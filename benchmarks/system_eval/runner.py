"""Resumable profile-isolated execution through the public v2 chat API."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import httpx

from .artifacts import atomic_json, digest, read_json
from .contracts import CONTRACT_VERSION, GENERATOR_CONTRACTS, PIPELINE_CONTRACTS, MatrixCell, expand_matrix, parse_generators, parse_pipelines
from .metrics import (
    binary_scores,
    factoid_scores,
    list_scores,
    ranking_metrics,
    rouge_n,
    rouge_su4,
    truncate_words,
    unjudged_rate_at,
    yesno_prediction,
)
from .reporting import generate_report
from .statistics import bootstrap_ci, holm_adjust, sign_randomization_pvalue, verdict


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    query: str
    gains: dict[str, int]
    question_type: str | None = None
    exact_answer: Any = None
    ideal_answer: str | None = None
    gold_snippets: list[str] | None = None


def normalise_document_id(value: Any) -> str:
    """Match numeric PMC document IDs across stored chunks and official qrels."""
    text = str(value or "").strip()
    upper = text.upper()
    if upper.startswith("PMCID:"):
        text = text.split(":", 1)[1]
    if text.upper().startswith("PMC"):
        text = text[3:]
    return text.lstrip("0") or "0"


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


class ProfileLifecycle:
    """Starts a fresh API for each generator profile and validates /health."""

    def __init__(self, base_url: str, *, managed: bool = True):
        self.base_url = base_url.rstrip("/")
        self.managed = managed
        self.process: subprocess.Popen[str] | None = None

    def start(self, generator: str, dataset: str) -> dict[str, Any]:
        contract = GENERATOR_CONTRACTS[generator]
        environment = os.environ.copy()
        environment.update({
            "CLOUD_MODE": "true",
            "SYSTEM_EVAL_MODE": "true",
            "GENERATION_BACKEND": contract.backend,
            "FOUNDRY_MODEL": "gpt-5.4",
            "CLOUD_EMBEDDING": "embed-v-4-0",
            "CLOUD_RERANKER": "Cohere-rerank-v4.0-fast",
            "DISABLE_DATASET_PRELOAD": "true",
            "SYSTEM_EVAL_DATASET": dataset,
        })
        dataset_root = Path(environment.get("SYSTEM_EVAL_DATA_DIR", "data/system-eval")) / dataset
        environment.update({
            "HNSW_INDEX_PATH": str(dataset_root / "hnsw_index.bin"),
            "HNSW_MAPPING_PATH": str(dataset_root / "hnsw_mapping.sqlite"),
            "LEXICAL_FTS_PATH": str(dataset_root / "bm25.sqlite"),
        })
        if generator == "medswin":
            local_health_url = (environment.get("MEDSWIN_LLM_URL") or "http://127.0.0.1:8000/v1/chat/completions").rsplit("/v1/", 1)[0] + "/health"
            local = httpx.get(local_health_url, timeout=10).json()
            if local.get("model") != contract.model or not str(local.get("model_revision") or "").strip():
                raise RuntimeError("local MedSwin server model/revision does not match the benchmark contract")
            configured_revision = environment.get("MEDSWIN_MODEL_REVISION")
            if configured_revision and configured_revision != local.get("model_revision"):
                raise RuntimeError("local MedSwin server revision does not match MEDSWIN_MODEL_REVISION")
            environment["MEDSWIN_MODEL_REVISION"] = str(local["model_revision"])
        if self.managed:
            try:
                existing = httpx.get(f"{self.base_url}/health", timeout=1)
                if existing.is_success:
                    raise RuntimeError(
                        f"Refusing to reuse an existing API at {self.base_url}; system-eval requires a fresh profile process"
                    )
            except httpx.HTTPError:
                pass
            port = self.base_url.rsplit(":", 1)[-1]
            self.process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port],
                env=environment,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        deadline = time.monotonic() + 90
        last_error = "not ready"
        while time.monotonic() < deadline:
            try:
                health = httpx.get(f"{self.base_url}/health", timeout=3).json()
                self._validate(health, generator)
                return health
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(1)
        self.stop()
        raise RuntimeError(f"API profile {generator} did not become ready: {last_error}")

    def _validate(self, health: dict[str, Any], generator: str) -> None:
        contract = GENERATOR_CONTRACTS[generator]
        expected = {
            "contract_version": CONTRACT_VERSION,
            "cloud_mode": True,
            "generation_backend": contract.backend,
            "generation_model": contract.model,
            "embedding_model": contract.embedding_model,
            "reranker_model": contract.reranker_model,
            "system_eval_mode": True,
        }
        for key, value in expected.items():
            if health.get(key) != value:
                raise RuntimeError(f"profile mismatch for {generator}: {key}={health.get(key)!r}, expected {value!r}")
        if generator == "medswin" and not str(health.get("resolved_local_model_revision") or "").strip():
            raise RuntimeError("local MedSwin API health does not expose its resolved model revision")
        if sorted(health.get("supported_pipelines") or []) != ["full", "naive", "two-stage"]:
            raise RuntimeError("profile does not expose the v2 three-pipeline contract")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def _case_order(cases: Iterable[Case], replicate: int) -> list[Case]:
    return sorted(cases, key=lambda case: sha256(f"{replicate}\0{case.case_id}".encode()).hexdigest())


def _load_cases(run_dir: Path, dataset: str, *, smoke: bool) -> list[Case]:
    source = run_dir / "datasets" / f"{dataset}.json"
    if not source.exists():
        return []
    rows = read_json(source)
    if smoke:
        wanted = ("diagnosis", "test", "treatment") if dataset == "trec" else ("yesno", "factoid", "list", "summary")
        selected = []
        for question_type in wanted:
            match = next((row for row in rows if row.get("question_type") == question_type), None)
            if match is None:
                raise RuntimeError(f"Smoke manifest has no {question_type} {dataset} case")
            selected.append(match)
        rows = selected
    return [Case(**row) for row in rows]


class SystemEvalRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cells = expand_matrix(parse_pipelines(args.pipeline), parse_generators(args.generator), custom=args.custom)
        root = Path(args.artifact_dir)
        suffix = args.run_id or digest({"matrix": [asdict(cell) for cell in self.cells], "preset": args.custom, "smoke": args.smoke})[:16]
        self.run_dir = root / suffix
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "raw", "rankings", "answers", "metrics", "statistics", "tables", "datasets"):
            (self.run_dir / name).mkdir(exist_ok=True)

    def prepare(self) -> None:
        from benchmarks.bioasq13b.prepare import prepare_bioasq
        from benchmarks.trec_cds2016.topics import load_topics

        datasets = self.run_dir / "datasets"
        datasets.mkdir(exist_ok=True)
        trec_cases = []
        qrels_path = Path("benchmarks/trec_cds2016/nist/qrels-treceval-2016.txt")
        gains: dict[str, dict[str, int]] = {}
        if qrels_path.exists():
            for line in qrels_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    gains.setdefault(parts[0], {})[normalise_document_id(parts[2])] = int(parts[3])
        for topic in load_topics():
            query = topic.system_eval_query()
            trec_cases.append(asdict(Case("trec", str(topic.number), query, gains.get(str(topic.number), {}), topic.topic_type)))
        atomic_json(datasets / "trec.json", trec_cases)
        bioasq = prepare_bioasq(cache_dir=Path(self.args.bioasq_cache_dir), credentials_required=not self.args.smoke)
        atomic_json(datasets / "bioasq.json", bioasq)
        bioasq_manifest = Path(self.args.bioasq_cache_dir) / "sample.manifest.json"
        self._write_manifest({
            "prepared": True,
            "trec_cases": len(trec_cases),
            "bioasq_cases": len(bioasq),
            "dataset_hashes": {
                "trec_cases": digest(trec_cases),
                "trec_qrels": sha256(qrels_path.read_bytes()).hexdigest() if qrels_path.exists() else None,
                "bioasq_sample_manifest": read_json(bioasq_manifest) if bioasq_manifest.exists() else None,
            },
        })

    def run(self) -> None:
        lifecycle = ProfileLifecycle(self.args.base_url, managed=True)
        datasets = [("trec", _load_cases(self.run_dir, "trec", smoke=self.args.smoke)), ("bioasq", _load_cases(self.run_dir, "bioasq", smoke=self.args.smoke))]
        if not any(cases for _, cases in datasets):
            raise RuntimeError("No prepared datasets; run --phase prepare first")
        generators = list(dict.fromkeys(cell.generator for cell in self.cells))
        profile_healths: dict[str, list[dict[str, Any]]] = {}
        for replicate in range(1, self.args.replicates + 1):
            if replicate % 2 == 0:
                generators.reverse()
            for generator in generators:
                for dataset, cases in datasets:
                    health = lifecycle.start(generator, dataset)
                    profile_healths.setdefault(generator, []).append(health)
                    try:
                        self._validate_dataset_readiness(dataset)
                        selected = [cell for cell in self.cells if cell.generator == generator]
                        if replicate % 2 == 0:
                            selected.reverse()
                        for case in _case_order(cases, replicate):
                            for cell in selected:
                                self._run_unit(case, cell, replicate, health)
                    finally:
                        lifecycle.stop()
        self._write_manifest({"run_completed": True, "profile_healths": profile_healths})

    def _validate_dataset_readiness(self, dataset: str) -> None:
        response = httpx.get(
            f"{self.args.base_url.rstrip('/')}/api/v1/storage/stats",
            params={"org_id": f"system-eval-{dataset}"},
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        stats = response.json()
        if int(stats.get("source_counts", {}).get("LIT", 0)) <= 0 or int(stats.get("active_embeddings", 0)) <= 0:
            raise RuntimeError(f"{dataset} benchmark corpus is not populated with active LIT embeddings")
        if not stats.get("index_exists") or not stats.get("index_provenance_valid"):
            raise RuntimeError(f"{dataset} benchmark index is missing or has invalid provenance")

    def _run_unit(self, case: Case, cell: MatrixCell, replicate: int, health: dict[str, Any]) -> None:
        path = self.run_dir / "raw" / case.dataset / cell.generator / cell.pipeline / str(replicate) / f"{case.case_id}.json"
        if path.exists():
            return
        payload = {
            "query": case.query,
            "user_id": f"system-eval-{case.dataset}",
            "org_id": f"system-eval-{case.dataset}",
            "session_id": f"{case.dataset}-{case.case_id}-{replicate}-{cell.generator}-{cell.pipeline}",
            "constraints": {"source_policy": "LIT_ONLY", "ranking_depth": 20},
            "benchmark": {"dataset": case.dataset, "case_id": case.case_id, "question_type": case.question_type, "response_schema": "bioasq"},
        }
        attempts, failure = [], None
        response: dict[str, Any] | None = None
        trace: dict[str, Any] | None = None
        for attempt in range(4):
            started = time.perf_counter()
            try:
                result = httpx.post(f"{self.args.base_url.rstrip('/')}/api/v1/pipelines/{cell.pipeline}/chat", json=payload, timeout=self.args.timeout)
                body = result.json()
                attempts.append({"attempt": attempt + 1, "status": result.status_code, "elapsed_ms": (time.perf_counter() - started) * 1000})
                if result.status_code < 400 and isinstance(body, dict) and body.get("ranked_evidence") is not None and body.get("answer_artifact") is not None:
                    trace = self._validate_response(body, cell, payload)
                    response = body
                    break
                failure = f"HTTP {result.status_code}: malformed structured response"
            except (httpx.HTTPError, ValueError) as exc:
                failure = str(exc)
                attempts.append({"attempt": attempt + 1, "error": failure, "elapsed_ms": (time.perf_counter() - started) * 1000})
                time.sleep(min(4, attempt + 1))
        atomic_json(path, {"case": asdict(case), "cell": asdict(cell), "replicate": replicate, "health": health, "request": payload, "response": response, "trace": trace, "attempts": attempts, "degraded": response is None or bool((response or {}).get("degraded_mode")), "failure": failure})
        atomic_json(self.run_dir / "checkpoints" / f"{case.dataset}-{case.case_id}-{cell.generator}-{cell.pipeline}-{replicate}.json", {"raw_path": str(path.relative_to(self.run_dir)), "completed": response is not None})
        if response:
            atomic_json(self.run_dir / "rankings" / f"{case.dataset}-{case.case_id}-{cell.generator}-{cell.pipeline}-{replicate}.json", response.get("ranked_evidence") or [])
            atomic_json(self.run_dir / "answers" / f"{case.dataset}-{case.case_id}-{cell.generator}-{cell.pipeline}-{replicate}.json", response.get("answer_artifact") or {})

    def _validate_response(self, response: dict[str, Any], cell: MatrixCell, payload: dict[str, Any]) -> dict[str, Any]:
        if response.get("pipeline") != cell.pipeline:
            raise RuntimeError(f"response pipeline mismatch: {response.get('pipeline')!r}")
        if response.get("generator_profile") != cell.generator:
            raise RuntimeError(f"response generator mismatch: {response.get('generator_profile')!r}")
        ranking = response.get("ranked_evidence") or []
        if len(ranking) > 20:
            raise RuntimeError("benchmark response exceeded top-20 ranking contract")
        trace_id = response.get("trace_id")
        if not trace_id:
            raise RuntimeError("benchmark response did not return a trace id")
        trace = httpx.get(
            f"{self.args.base_url.rstrip('/')}/api/v1/pipelines/traces/{trace_id}",
            params={"org_id": payload["org_id"], "include_details": "true"},
            timeout=self.args.timeout,
        ).json()
        calls = [row.get("tool_name") for row in trace.get("tool_calls", [])]
        if cell.pipeline == "naive" and any(name in {"retrieval.rerank", "pipeline.mac"} or str(name).startswith("agent.") for name in calls):
            raise RuntimeError("naive trace contains an unsupported retrieval/MAC stage")
        if cell.pipeline == "two-stage":
            if "retrieval.rerank" not in calls or "pipeline.mac" not in calls:
                raise RuntimeError("two-stage trace is missing reranking or the explicit MAC-skip event")
            if any(str(name).startswith("agent.") for name in calls):
                raise RuntimeError("two-stage trace invoked MAC")
        if cell.pipeline == "full" and not any(str(name).startswith("agent.") for name in calls):
            raise RuntimeError("full trace did not invoke MAC")
        return trace

    def score(self) -> None:
        records: list[dict[str, Any]] = []
        claim_judge = None
        for path in sorted((self.run_dir / "raw").glob("*/*/*/*/*.json")):
            raw = read_json(path)
            response, case = raw.get("response"), raw["case"]
            if not response:
                continue
            ranking = [normalise_document_id(item.get("doc_id")) for item in response.get("ranked_evidence") or []]
            gains = case.get("gains") or {}
            row = {"dataset": case["dataset"], "case_id": case["case_id"], **raw["cell"], "replicate": raw["replicate"], "degraded": raw["degraded"], **ranking_metrics(ranking, gains)}
            ranked_chunks = [item.get("winning_chunk_id") for item in response.get("ranked_evidence") or []]
            row["duplicate_chunk_collapse_count"] = max(0, len(ranked_chunks) - len(set(ranked_chunks)))
            if case["dataset"] == "trec":
                for cutoff in (5, 10, 20):
                    row[f"unjudged_rate@{cutoff}"] = unjudged_rate_at(ranking, set(gains), cutoff)
            artifact = response.get("answer_artifact") or {}
            ideal = case.get("ideal_answer") or ""
            answer = truncate_words(artifact.get("ideal_answer") or artifact.get("content") or "")
            if case["dataset"] == "bioasq":
                row.update({f"rouge2_{name}": value for name, value in rouge_n(answer, ideal, 2).items()})
                row.update({f"rouge_su4_{name}": value for name, value in rouge_su4(answer, ideal).items()})
                row["abstained"] = bool(artifact.get("abstained"))
                if row["abstained"]:
                    row.update({"rouge2_recall": 0.0, "rouge2_f1": 0.0, "rouge_su4_recall": 0.0, "rouge_su4_f1": 0.0})
                question_type = case.get("question_type")
                exact = artifact.get("exact_answer")
                if question_type == "yesno":
                    predicted, expected = yesno_prediction(exact), yesno_prediction(case.get("exact_answer"))
                    if predicted is not None and expected is not None:
                        row.update({f"yesno_{key}": value for key, value in binary_scores(predicted, expected).items()})
                elif question_type == "factoid":
                    row.update({f"factoid_{key}": value for key, value in factoid_scores(exact, case.get("exact_answer")).items()})
                elif question_type == "list":
                    predicted = exact if isinstance(exact, list) else [exact] if exact else []
                    expected = case.get("exact_answer") if isinstance(case.get("exact_answer"), list) else [case.get("exact_answer")]
                    row.update({f"list_{key}": value for key, value in list_scores(predicted, expected).items()})
                row["answer_coverage"] = float(bool(answer))
                if not row["abstained"]:
                    if claim_judge is None:
                        from .claims import LocalNliJudge
                        claim_judge = LocalNliJudge()
                    from .claims import claim_metrics, split_claims
                    cited = [
                        passage.get("text", "")
                        for passage in (response.get("evidence_bundle") or {}).get("passages", [])
                        if passage.get("chunk_id") in set(artifact.get("cited_chunk_ids") or [])
                    ]
                    row.update(claim_metrics(split_claims(answer), split_claims(ideal), cited, claim_judge))
            timing = response.get("timing_ms") or {}
            row.update({f"timing_{key}_ms": value for key, value in timing.items() if isinstance(value, (int, float))})
            calls = ((raw.get("trace") or {}).get("tool_calls") or [])
            row["reranker_call_count"] = sum(call.get("tool_name") == "retrieval.rerank" for call in calls)
            row["llm_call_count"] = sum(str(call.get("tool_name")).startswith("agent.") for call in calls) + 1
            row["retry_count"] = max(0, len(raw.get("attempts") or []) - 1)
            records.append(row)
        atomic_json(self.run_dir / "metrics" / "records.json", records)
        self._write_manifest({"score_completed": True, "metric_records": len(records)})

    def report(self) -> None:
        records = read_json(self.run_dir / "metrics" / "records.json") if (self.run_dir / "metrics" / "records.json").exists() else []
        clean = [row for row in records if not row.get("degraded")]
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in clean:
            grouped.setdefault((row["dataset"], row["generator"], row["pipeline"]), []).append(row)
        primary = []
        for (dataset, generator, pipeline), rows in sorted(grouped.items()):
            endpoint = "ndcg@10" if dataset == "trec" else "rouge_su4_f1"
            primary.append({"dataset": dataset, "generator": generator, "pipeline": pipeline, "endpoint": endpoint, "mean": round(sum(row.get(endpoint, 0.0) for row in rows) / len(rows), 6), "clean_n": len(rows)})
        effects = self._adjacent_effects(clean)
        report = {
            "contract_version": CONTRACT_VERSION,
            "run_dir": str(self.run_dir),
            "primary_endpoints": primary,
            "stage_effects": effects,
            "retrieval": records,
            "answers": [row for row in records if row["dataset"] == "bioasq"],
            "operational": self._operational(records),
            "limitations": ["LIT-only scope", "incomplete TREC judgments", "derived BioASQ corpus", "automated NLI diagnostics", "old-dataset contamination risk", "no clinician validation"],
        }
        generate_report(self.run_dir, report, smoke=self.args.smoke)
        self._write_manifest({"report_completed": True})

    @staticmethod
    def _adjacent_effects(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_case: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in records:
            by_case.setdefault((row["dataset"], row["generator"], row["pipeline"], row["case_id"]), []).append(row)
        tests: list[dict[str, Any]] = []
        for generator in ("cloud", "medswin"):
            for left, right in (("naive", "two-stage"), ("two-stage", "full")):
                for dataset, metric in (("trec", "ndcg@10"), ("bioasq", "rouge_su4_f1")):
                    left_cases = {key[3]: values for key, values in by_case.items() if key[:3] == (dataset, generator, left)}
                    right_cases = {key[3]: values for key, values in by_case.items() if key[:3] == (dataset, generator, right)}
                    shared = sorted(set(left_cases) & set(right_cases))
                    deltas = []
                    for case_id in shared:
                        lrep = {row["replicate"]: row for row in left_cases[case_id]}
                        rrep = {row["replicate"]: row for row in right_cases[case_id]}
                        matched = sorted(set(lrep) & set(rrep))
                        if matched:
                            deltas.append(sum(rrep[rep][metric] - lrep[rep][metric] for rep in matched) / len(matched))
                    coverage = len(deltas) / (30 if dataset == "trec" else 120)
                    mean = sum(deltas) / len(deltas) if deltas else float("nan")
                    ci = bootstrap_ci(deltas, seed=f"{dataset}:{generator}:{left}:{right}:{metric}")
                    p = sign_randomization_pvalue(deltas, seed=f"{dataset}:{generator}:{left}:{right}:{metric}")
                    tests.append({"generator": generator, "transition": f"{left}_to_{right}", "dataset": dataset, "metric": metric, "mean_delta": mean, "ci_low": ci[0], "ci_high": ci[1], "p_raw": p, "coverage": coverage})
        corrected = holm_adjust([row["p_raw"] for row in tests]) if tests else []
        for row, adjusted in zip(tests, corrected):
            row["p_holm"] = adjusted
        by_transition: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for row in tests:
            by_transition.setdefault((row["generator"], row["transition"]), {})[row["dataset"]] = row
        for key, cells in by_transition.items():
            trec, bioasq = cells.get("trec"), cells.get("bioasq")
            if trec and bioasq:
                label = verdict(retrieval_delta=trec["mean_delta"], answer_delta=bioasq["mean_delta"], retrieval_ci=(trec["ci_low"], trec["ci_high"]), answer_ci=(bioasq["ci_low"], bioasq["ci_high"]), retrieval_p=trec["p_holm"], answer_p=bioasq["p_holm"], coverage=min(trec["coverage"], bioasq["coverage"]))
                trec["stage_verdict"] = bioasq["stage_verdict"] = label
        return tests

    @staticmethod
    def _operational(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault((row["dataset"], row["generator"], row["pipeline"]), []).append(row)
        output = []
        for (dataset, generator, pipeline), rows in sorted(groups.items()):
            latency = sorted(float(row.get("timing_total_ms", 0.0)) for row in rows)
            if latency:
                p95 = latency[min(len(latency) - 1, max(0, int((len(latency) - 1) * 0.95)))]
            else:
                p95 = 0.0
            output.append({
                "dataset": dataset,
                "generator": generator,
                "pipeline": pipeline,
                "observations": len(rows),
                "latency_p50_ms": latency[len(latency) // 2] if latency else 0.0,
                "latency_p95_ms": p95,
                "mean_reranker_calls": sum(row.get("reranker_call_count", 0) for row in rows) / len(rows),
                "mean_llm_calls": sum(row.get("llm_call_count", 0) for row in rows) / len(rows),
                "mean_retries": sum(row.get("retry_count", 0) for row in rows) / len(rows),
                "abstention_rate": sum(bool(row.get("abstained")) for row in rows) / len(rows),
                "degradation_rate": sum(bool(row.get("degraded")) for row in rows) / len(rows),
            })
        return output

    def _write_manifest(self, updates: dict[str, Any]) -> None:
        path = self.run_dir / "manifest.json"
        current = read_json(path) if path.exists() else {}
        current.update(updates)
        current.setdefault("run_id", self.run_dir.name)
        current.setdefault("contract_version", CONTRACT_VERSION)
        current.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        current["preset"] = self.args.custom
        current["matrix"] = [asdict(cell) for cell in self.cells]
        current["replicates"] = self.args.replicates
        current["command"] = "system-eval"
        current["platform"] = platform.platform()
        current["pipeline_contracts"] = {name: asdict(value) for name, value in PIPELINE_CONTRACTS.items()}
        current["generator_contracts"] = {name: asdict(value) for name, value in GENERATOR_CONTRACTS.items()}
        current.setdefault("git", self._git_metadata())
        current.setdefault("runtime", {"python": sys.version, "platform": platform.platform()})
        current.setdefault("retry_policy", {"attempts": 4, "degraded_excluded_from_quality": True})
        atomic_json(path, current)

    @staticmethod
    def _git_metadata() -> dict[str, Any]:
        def command(*parts: str) -> str:
            result = subprocess.run(parts, text=True, capture_output=True, check=False)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        return {"sha": command("git", "rev-parse", "HEAD"), "dirty": bool(command("git", "status", "--porcelain"))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system-eval")
    parser.add_argument("--phase", choices=("prepare", "run", "score", "report", "all"), default="all")
    parser.add_argument("--pipeline", default="all")
    parser.add_argument("--generator", default="all")
    parser.add_argument("--custom")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--artifact-dir", default=os.getenv("SYSTEM_EVAL_ARTIFACT_DIR", "artifacts/system-eval"))
    parser.add_argument("--bioasq-cache-dir", default=os.getenv("BIOASQ_CACHE_DIR", "data/system-eval/bioasq13b"))
    parser.add_argument("--base-url", default=os.getenv("SYSTEM_EVAL_BASE_URL", "http://127.0.0.1:8110"))
    parser.add_argument("--run-id")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.custom and (args.pipeline != "all" or args.generator != "all"):
        raise SystemExit("--custom is mutually exclusive with --pipeline and --generator")
    if args.smoke:
        args.replicates = 1
    runner = SystemEvalRunner(args)
    phases = ("prepare", "run", "score", "report") if args.phase == "all" else (args.phase,)
    for phase in phases:
        getattr(runner, phase)()
    print(runner.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
