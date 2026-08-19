#!/usr/bin/env python3
"""Run the strict 2x2 complete-TREC MedSwin evaluation matrix.

Cells:
  1. naive_rag + local MedSwin-DaRE-TIES-KD-0.7
  2. naive_rag + Azure Foundry GPT-5.4
  3. full MedSwin (hybrid + Cohere rerank + MAC + gate) + local MedSwin 7B
  4. full MedSwin + Azure Foundry GPT-5.4

The corpus/index is prepared exactly once by ``prepare_full_trec_runtime.py``.
Every API process uses Cohere Embed v4 query embeddings against that same
``input_type=document`` index. The runner fails a cell when a naive request
falls back from ANN, when the full system silently skips/fails reranking, or
when the expected MAC specialists did not actually execute.

Generator comparison uses one shared context/output envelope for both local
MedSwin and GPT-5.4. This prevents the cloud model's larger context window from
receiving more evidence merely because of backend capacity; the local server
fails rather than silently truncating if the shared envelope is still too large.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
for root in (REPO_ROOT, EVAL_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.database import get_sync_database
from eval.app.audit import aggregate_run, audit_case
from eval.app.client import MedSwinClient
from eval.app.schemas import BenchmarkCase, RunAudit

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_MAC_AGENTS = {"emr", "guideline", "safety", "quality", "critic"}
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"
DEFAULT_CASES = EVAL_ROOT / "data" / "trec-cds-2016" / "full" / "cases.jsonl"

# These are intentionally conservative defaults for a LLaMA-family 7B context.
# All four cells receive the same values. They remain environment-overridable
# and are recorded in every audit so a paper can state the exact envelope.
DEFAULT_SHARED_TOKEN_BUDGET = 700
DEFAULT_AGENT_PASSAGE_LIMIT = 4
DEFAULT_AGENT_PASSAGE_MAX_CHARS = 500
DEFAULT_GENERATION_MAX_TOKENS = 384


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> list[BenchmarkCase]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(BenchmarkCase.model_validate(json.loads(line)))
    if len(cases) != EXPECTED_QUERIES:
        raise RuntimeError(f"Full matrix requires {EXPECTED_QUERIES} cases; found {len(cases)}")
    return cases


def _validate_runtime_manifest(org_id: str) -> dict[str, Any]:
    path = MANIFEST_DIR / f"full-trec-runtime-{org_id}.json"
    if not path.exists():
        raise RuntimeError(f"Missing {path}; run eval/scripts/prepare_full_trec_runtime.py first")
    manifest = _load_json(path)
    checks = {
        "complete": manifest.get("complete") is True,
        "dataset": manifest.get("dataset") == EXPECTED_DATASET,
        "documents": manifest.get("expected_documents") == EXPECTED_DOCS,
        "queries": manifest.get("queries") == EXPECTED_QUERIES,
        "embedding_model": manifest.get("embedding_model") == "embed-v-4-0",
        "embedding_input_type": manifest.get("embedding_input_type") == "document",
        "no_stale_chunks": int((manifest.get("mongo") or {}).get("stale_chunks") or -1) == 0,
    }
    total_chunks = int((manifest.get("mongo") or {}).get("total_chunks") or 0)
    checks["chunks_nonzero"] = total_chunks > 0
    checks["bm25_complete"] = int((manifest.get("bm25") or {}).get("rows") or 0) == total_chunks
    checks["hnsw_complete"] = int((manifest.get("hnsw") or {}).get("total_vectors") or 0) == total_chunks
    checks["all_active"] = int((manifest.get("mongo") or {}).get("active_chunks") or 0) == total_chunks
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Full runtime manifest failed: " + ", ".join(failed))

    for field in (
        (manifest.get("bm25") or {}).get("path"),
        (manifest.get("hnsw") or {}).get("index_path"),
        (manifest.get("hnsw") or {}).get("mapping_path"),
    ):
        if not field or not Path(field).exists():
            raise RuntimeError(f"Required full-runtime artifact is missing: {field}")
    return manifest


def _qrel_presence(cases: list[BenchmarkCase], org_id: str) -> set[str]:
    gold = {doc_id for case in cases for doc_id in case.gold_doc_ids}
    if not gold:
        raise RuntimeError("Full TREC cases contain no positive qrels")
    coll = get_sync_database()["chunks"]
    present = set(
        str(value)
        for value in coll.distinct(
            "doc_id",
            {"org_id": org_id, "source_type": "LIT", "doc_id": {"$in": list(gold)}},
        )
    )
    if present != gold:
        missing = sorted(gold - present)
        raise RuntimeError(f"{len(missing)} positive qrel documents are absent; sample={missing[:20]}")
    return present


def _trace_document(trace_id: str, org_id: str) -> dict[str, Any] | None:
    return get_sync_database()["traces"].find_one({"trace_id": trace_id, "org_id": org_id}, {"_id": 0})


def _degraded(response: dict[str, Any]) -> bool:
    value = response.get("degraded_mode")
    if value is True:
        return True
    return isinstance(value, dict) and any(bool(item) for item in value.values())


def _strict_naive_errors(response: dict[str, Any], trace: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if response.get("pipeline") != "naive_rag":
        errors.append(f"unexpected_pipeline:{response.get('pipeline')}")
    if response.get("retrieval_backend") != "ann":
        errors.append(f"naive_not_ann:{response.get('retrieval_backend')}")
    if _degraded(response):
        errors.append("naive_degraded")
    if not str(response.get("answer") or "").strip():
        errors.append("empty_answer")
    if not (response.get("evidence_bundle") or {}).get("passages"):
        errors.append("naive_empty_evidence")
    if trace is None:
        errors.append("missing_trace")
    else:
        tools = {str(item.get("tool_name")) for item in trace.get("tool_calls") or []}
        if "retrieval.naive_dense" not in tools:
            errors.append("missing_naive_dense_trace")
        if trace.get("rerank_traces"):
            errors.append("naive_used_reranker")
        agent_tools = [name for name in tools if name.startswith("agent.")]
        if agent_tools:
            errors.append("naive_used_mac")
        if trace.get("sufficiency_checks"):
            errors.append("naive_used_sufficiency_gate")
    return errors


def _strict_full_errors(response: dict[str, Any], trace: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if response.get("pipeline") != "medswin":
        errors.append(f"unexpected_pipeline:{response.get('pipeline')}")
    if _degraded(response):
        errors.append("medswin_degraded")
    if not str(response.get("answer") or "").strip():
        errors.append("empty_answer")
    if not response.get("policy_decision"):
        errors.append("missing_policy_decision")
    if not response.get("sufficiency_decision"):
        errors.append("missing_sufficiency_decision")
    if not response.get("facet_matrix"):
        errors.append("missing_facet_matrix")
    if trace is None:
        errors.append("missing_trace")
        return errors

    retrieval_traces = trace.get("retrieval_traces") or []
    rerank_traces = trace.get("rerank_traces") or []
    sufficiency_checks = trace.get("sufficiency_checks") or []
    tools = trace.get("tool_calls") or []
    if not retrieval_traces:
        errors.append("missing_hybrid_retrieval_trace")
    if not rerank_traces:
        errors.append("missing_rerank_trace")
    if not sufficiency_checks:
        errors.append("missing_sufficiency_checks")

    rerank_tools = [item for item in tools if item.get("tool_name") == "retrieval.rerank"]
    if not rerank_tools:
        errors.append("reranker_not_executed")
    for item in rerank_tools:
        result = item.get("result") or {}
        version = str(result.get("calibration_version") or "")
        if "rerank-error" in version or int(result.get("n") or 0) <= 0:
            errors.append("reranker_failed_open")
            break
    for item in rerank_traces:
        version = str(item.get("calibration_version") or "")
        if "rerank-error" in version:
            errors.append("reranker_failed_open")
            break
        if not item.get("scores"):
            errors.append("reranker_returned_no_scores")
            break

    agent_calls = [item for item in tools if str(item.get("tool_name") or "").startswith("agent.")]
    agent_names = {str(item.get("tool_name")).split(".", 1)[1] for item in agent_calls}
    missing_agents = sorted(EXPECTED_MAC_AGENTS - agent_names)
    if missing_agents:
        errors.append("mac_agents_missing:" + ",".join(missing_agents))
    for item in agent_calls:
        result = item.get("result") or {}
        if result.get("degraded") or result.get("error"):
            errors.append(f"mac_agent_failed:{item.get('tool_name')}")
    message_agents = {
        str(item.get("agent_id"))
        for item in trace.get("messages") or []
        if item.get("agent_id")
    }
    if not EXPECTED_MAC_AGENTS.issubset(message_agents):
        errors.append("mac_message_provenance_incomplete")
    return sorted(set(errors))


def _trace_summary(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    bundle = trace.get("evidence_bundle") or {}
    return {
        "messages_count": len(trace.get("messages") or []),
        "tool_calls_count": len(trace.get("tool_calls") or []),
        "sufficiency_checks_count": len(trace.get("sufficiency_checks") or []),
        "evidence_passages_count": len(bundle.get("passages") or []),
    }


def _health_url(completions_url: str) -> str:
    parsed = urlsplit(completions_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _wait_http(url: str, process: subprocess.Popen[Any] | None, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Process exited before health check {url}: code={process.returncode}")
            try:
                response = client.get(url)
                if response.is_success:
                    return response.json()
                last_error = RuntimeError(f"HTTP {response.status_code}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Health check timed out for {url}: {last_error}")


def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:  # noqa: BLE001
        process.kill()
        process.wait(timeout=10)


def _start_local_medswin() -> tuple[subprocess.Popen[Any] | None, dict[str, Any]]:
    completions_url = os.getenv("MEDSWIN_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
    health_url = _health_url(completions_url)
    try:
        health = _wait_http(health_url, None, 2.0)
        if health.get("model") != "MedSwin/MedSwin-DaRE-TIES-KD-0.7":
            raise RuntimeError(f"Port already serves the wrong model: {health.get('model')}")
        if not health.get("context_window"):
            raise RuntimeError("Existing MedSwin server does not report its context window")
        return None, health
    except Exception:
        pass

    log_path = REPO_ROOT / "logs" / "full-eval-medswin-llm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "serve-medswin-llm.py")],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    health = _wait_http(
        health_url,
        process,
        float(os.getenv("MEDSWIN_LLM_STARTUP_TIMEOUT_S", "900")),
    )
    if health.get("model") != "MedSwin/MedSwin-DaRE-TIES-KD-0.7":
        _terminate(process)
        raise RuntimeError(f"Local generator health reported unexpected model {health.get('model')}")
    if not health.get("context_window"):
        _terminate(process)
        raise RuntimeError("Local MedSwin server did not report a context window")
    return process, health


def _shared_eval_envelope() -> dict[str, int]:
    values = {
        "token_budget": int(os.getenv("FULL_EVAL_TOKEN_BUDGET_B", str(DEFAULT_SHARED_TOKEN_BUDGET))),
        "agent_passage_limit": int(os.getenv("FULL_EVAL_AGENT_PASSAGE_LIMIT", str(DEFAULT_AGENT_PASSAGE_LIMIT))),
        "agent_passage_max_chars": int(
            os.getenv("FULL_EVAL_AGENT_PASSAGE_MAX_CHARS", str(DEFAULT_AGENT_PASSAGE_MAX_CHARS))
        ),
        "generation_max_tokens": int(
            os.getenv("FULL_EVAL_GENERATION_MAX_TOKENS", str(DEFAULT_GENERATION_MAX_TOKENS))
        ),
    }
    if any(value <= 0 for value in values.values()):
        raise RuntimeError(f"Invalid full-eval shared prompt envelope: {values}")
    return values


def _api_env(generator: str, port: int, org_id: str) -> dict[str, str]:
    env = os.environ.copy()
    envelope = _shared_eval_envelope()
    env.update(
        {
            "CLOUD_MODE": "true",
            "CLOUD_EMBEDDING": "embed-v-4-0",
            "CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE": "query",
            "CLOUD_RERANKER": "Cohere-rerank-v4.0-fast",
            "GENERATION_BACKEND": generator,
            "FOUNDRY_MODEL": os.getenv("FOUNDRY_MODEL", "gpt-5.4"),
            "CLOUD_MODEL": os.getenv("FOUNDRY_MODEL", "gpt-5.4"),
            "APP_PORT": str(port),
            "BENCHMARK_ORG_ID": org_id,
            "MEDSWIN_BASE_URL": f"http://127.0.0.1:{port}",
            "NAIVE_ENABLE_MONGO_FALLBACK": "false",
            "TOKEN_BUDGET_B": str(envelope["token_budget"]),
            "AGENT_PASSAGE_LIMIT": str(envelope["agent_passage_limit"]),
            "AGENT_PASSAGE_MAX_CHARS": str(envelope["agent_passage_max_chars"]),
            "LLM_DEFAULT_MAX_TOKENS": str(envelope["generation_max_tokens"]),
        }
    )
    return env


def _start_api(generator: str, port: int, org_id: str) -> subprocess.Popen[Any]:
    log_path = REPO_ROOT / "logs" / f"full-eval-api-{generator}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=_api_env(generator, port, org_id),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _wait_http(f"http://127.0.0.1:{port}/health", process, 180.0)
    return process


async def _run_cell(
    cases: list[BenchmarkCase],
    pipeline: str,
    generator: str,
    port: int,
    org_id: str,
    gold_present: set[str],
    top_k: int,
    timeout_s: float,
) -> RunAudit:
    run_id = f"full-{pipeline}-{generator}-{uuid.uuid4().hex[:10]}"
    envelope = _shared_eval_envelope()
    run = RunAudit(
        run_id=run_id,
        dataset=EXPECTED_DATASET,
        config={
            "pipeline": pipeline,
            "generator": generator,
            "generator_model": (
                "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
                if generator == "medswin_local"
                else os.getenv("FOUNDRY_MODEL", "gpt-5.4")
            ),
            "embedding_model": "embed-v-4-0",
            "embedding_input_type": "query",
            "reranker_model": "Cohere-rerank-v4.0-fast" if pipeline == "medswin" else None,
            "cases": len(cases),
            "top_k": top_k,
            "strict_full_corpus": True,
            "shared_generation_envelope": envelope,
        },
    )
    base_url = f"http://127.0.0.1:{port}"
    async with MedSwinClient(
        base_url,
        org_id,
        os.getenv("BENCHMARK_USER_ID", "bench-user"),
        timeout_s=timeout_s,
        include_patient_context_in_query=False,
    ) as client:
        naive_ready = await client.naive_ready()
        if not naive_ready.get("mongo"):
            raise RuntimeError(f"API Mongo preflight failed: {naive_ready}")
        for index, case in enumerate(cases, start=1):
            response: dict[str, Any]
            try:
                response = await client.chat(
                    case,
                    source_policy="ANY",
                    guideline_only=False,
                    min_evidence_grade=0.3,
                    clinical_scope="clinician_cds",
                    pipeline=pipeline,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                response = {
                    "answer": "",
                    "pipeline": pipeline,
                    "evidence_bundle": {},
                    "citations": [],
                    "degraded_mode": {"request_error": True},
                }
                strict_errors = [f"request_failed:{exc}"]
                trace = None
            else:
                trace_id = str(response.get("trace_id") or "")
                trace = _trace_document(trace_id, org_id) if trace_id else None
                strict_errors = (
                    _strict_naive_errors(response, trace)
                    if pipeline == "naive_rag"
                    else _strict_full_errors(response, trace)
                )
            case_audit = audit_case(
                case,
                response,
                _trace_summary(trace),
                errors=strict_errors,
                available_doc_ids=gold_present,
                indexed_doc_ids=gold_present,
                pipeline=pipeline,
            )
            run.cases.append(case_audit)
            print(
                f"[full-matrix] {generator}/{pipeline} case {index}/{len(cases)} "
                f"errors={len(strict_errors)} msas={case_audit.msas:.4f}",
                flush=True,
            )
    aggregate_run(run)
    run.diagnostics.update(
        {
            "strict_error_cases": sum(1 for case in run.cases if case.errors),
            "strict_pass": all(not case.errors for case in run.cases),
            "generator": generator,
            "pipeline": pipeline,
        }
    )
    return run


def _run_output_dir() -> Path:
    value = os.getenv("RUN_STORE_DIR", "/tmp/medswin-audits")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_run(run: RunAudit) -> Path:
    path = _run_output_dir() / f"{run.run_id}.json"
    path.write_text(json.dumps(run.model_dump(), indent=2, default=str), encoding="utf-8")
    return path


def _metric(run: RunAudit, name: str) -> float:
    return float((run.aggregate or {}).get(name) or 0.0)


async def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _validate_runtime_manifest(args.org_id)
    cases_path = Path(args.cases_path or manifest.get("cases_path") or DEFAULT_CASES)
    cases = _load_cases(cases_path)
    gold_present = _qrel_presence(cases, args.org_id)

    local_process, local_health = _start_local_medswin()
    results: dict[str, RunAudit] = {}
    try:
        for generator in ("medswin_local", "foundry"):
            api_process = _start_api(generator, args.api_port, args.org_id)
            try:
                for pipeline in ("naive_rag", "medswin"):
                    key = f"{pipeline}:{generator}"
                    run = await _run_cell(
                        cases,
                        pipeline,
                        generator,
                        args.api_port,
                        args.org_id,
                        gold_present,
                        args.top_k,
                        args.timeout,
                    )
                    results[key] = run
                    output = _save_run(run)
                    if not run.diagnostics.get("strict_pass"):
                        print(f"[full-matrix] STRICT FAILURE retained for audit: {output}", flush=True)
                    else:
                        print(f"[full-matrix] strict pass: {output}", flush=True)
            finally:
                _terminate(api_process)
                await asyncio.sleep(1.0)

            # If this runner owns the local 7B process, release its GPU memory
            # before launching the GPT-only cells. An externally managed server
            # is never killed by the benchmark runner.
            if generator == "medswin_local" and local_process is not None:
                _terminate(local_process)
                local_process = None
                await asyncio.sleep(1.0)
    finally:
        _terminate(local_process)

    matrix = {
        "created_at": _now(),
        "complete_trec_runtime": manifest,
        "local_medswin_health": local_health,
        "shared_generation_envelope": _shared_eval_envelope(),
        "cells": {
            key: {
                "run_id": run.run_id,
                "strict_pass": bool(run.diagnostics.get("strict_pass")),
                "aggregate": run.aggregate,
                "error_cases": int(run.diagnostics.get("strict_error_cases") or 0),
            }
            for key, run in results.items()
        },
        "comparisons": {},
    }
    for generator in ("medswin_local", "foundry"):
        naive = results[f"naive_rag:{generator}"]
        full = results[f"medswin:{generator}"]
        matrix["comparisons"][f"full_minus_naive:{generator}"] = {
            "mean_msas": _metric(full, "mean_msas") - _metric(naive, "mean_msas"),
            "mean_evidence_doc_recall": _metric(full, "mean_evidence_doc_recall") - _metric(naive, "mean_evidence_doc_recall"),
            "mean_groundedness_proxy": _metric(full, "mean_groundedness_proxy") - _metric(naive, "mean_groundedness_proxy"),
        }
    for pipeline in ("naive_rag", "medswin"):
        local = results[f"{pipeline}:medswin_local"]
        cloud = results[f"{pipeline}:foundry"]
        matrix["comparisons"][f"gpt54_minus_medswin7b:{pipeline}"] = {
            "mean_msas": _metric(cloud, "mean_msas") - _metric(local, "mean_msas"),
            "mean_groundedness_proxy": _metric(cloud, "mean_groundedness_proxy") - _metric(local, "mean_groundedness_proxy"),
        }
    matrix["strict_pass"] = all(cell["strict_pass"] for cell in matrix["cells"].values())
    path = _run_output_dir() / f"full-matrix-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
    matrix["matrix_path"] = str(path)
    if not matrix["strict_pass"]:
        raise RuntimeError(f"One or more full evaluation cells failed strict architecture validation; matrix={path}")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default=os.getenv("BENCHMARK_ORG_ID", "bench-org"))
    parser.add_argument("--cases-path", default="")
    parser.add_argument("--api-port", type=int, default=int(os.getenv("FULL_EVAL_API_PORT", "8110")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("FULL_EVAL_TOP_K", "5")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("FULL_EVAL_REQUEST_TIMEOUT_S", "600")))
    return parser.parse_args()


def main() -> int:
    matrix = asyncio.run(run_matrix(parse_args()))
    print(json.dumps({"strict_pass": matrix["strict_pass"], "matrix_path": matrix["matrix_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
