#!/usr/bin/env python3
"""Run the strict complete-TREC MedSwin evaluation matrix.

Default cells (`--pipeline both`):
  1. naive_rag + local MedSwin-DaRE-TIES-KD-0.7
  2. naive_rag + Azure Foundry GPT-5.4
  3. full MedSwin (dense ANN + BM25 -> Cohere rerank -> MAC -> gate) + local MedSwin 7B
  4. full MedSwin + Azure Foundry GPT-5.4

`--pipeline naive_rag` or `--pipeline medswin` runs only those retrieval cells
against both generators. That subset is still architecture-strict, but it is not
the publication 2x2 (`publication_complete` is false).

This runner is fail-closed and self-certifying. It re-verifies the persisted
complete-TREC runtime before starting any cell, binds every API subprocess to the
HNSW/SQLite-BM25 artifacts recorded by that verified runtime manifest, validates
the actual generator identity reported by the API, proves both systems obey the
same final evidence-token budget, and records TREC/qrel diagnostics from the
*pre-pack retrieval ranking* separately from the final RAG evidence packet.
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
SCRIPTS_ROOT = EVAL_ROOT / "scripts"
for root in (REPO_ROOT, EVAL_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.config import settings as app_settings
from app.core.database import get_sync_database
from eval.app.full_contract import (
    ALL_GENERATORS,
    expected_matrix_keys,
    is_publication_matrix,
    resolve_pipelines,
)
from eval.app.audit import aggregate_run, audit_case, ranked_trec_metrics
from eval.app.client import MedSwinClient
from eval.app.schemas import BenchmarkCase, RunAudit
from verify_full_trec_runtime import verify as verify_full_runtime

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_MAC_AGENTS = {"emr", "guideline", "safety", "quality", "critic"}
EXPECTED_LOCAL_MODEL = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
EXPECTED_EMBEDDING = "embed-v-4-0"
EXPECTED_RERANKER = "Cohere-rerank-v4.0-fast"
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"
DEFAULT_CASES = EVAL_ROOT / "data" / "trec-cds-2016" / "full" / "cases.jsonl"

# Conservative shared envelope for the 7B model. All four cells get exactly the
# same values so generator capacity does not change the evidence budget.
DEFAULT_SHARED_TOKEN_BUDGET = 700
DEFAULT_AGENT_PASSAGE_LIMIT = 4
DEFAULT_AGENT_PASSAGE_MAX_CHARS = 500
DEFAULT_GENERATION_MAX_TOKENS = 384
DEFAULT_RETRIEVAL_TOP_K = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(BenchmarkCase.model_validate(json.loads(line)))
    if len(cases) != EXPECTED_QUERIES:
        raise RuntimeError(f"Full matrix requires {EXPECTED_QUERIES} cases; found {len(cases)}")
    case_ids = {case.case_id for case in cases}
    if len(case_ids) != EXPECTED_QUERIES:
        raise RuntimeError("Full matrix case IDs are not unique")
    return cases


def _validate_runtime_manifest(org_id: str) -> dict[str, Any]:
    path = MANIFEST_DIR / f"full-trec-runtime-{org_id}.json"
    manifest = _load_json(path)
    checks = {
        "complete": manifest.get("complete") is True,
        "dataset": manifest.get("dataset") == EXPECTED_DATASET,
        "documents": int(manifest.get("expected_documents") or 0) == EXPECTED_DOCS,
        "queries": int(manifest.get("queries") or 0) == EXPECTED_QUERIES,
        "embedding_model": manifest.get("embedding_model") == EXPECTED_EMBEDDING,
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

    required = (
        (manifest.get("bm25") or {}).get("path"),
        (manifest.get("hnsw") or {}).get("index_path"),
        (manifest.get("hnsw") or {}).get("mapping_path"),
    )
    for field in required:
        if not field or not Path(str(field)).exists():
            raise RuntimeError(f"Required full-runtime artifact is missing: {field}")
    return manifest


def _configure_parent_for_verification(manifest: dict[str, Any]) -> None:
    """Make independent verification use the runtime's exact embedding space."""
    app_settings.CLOUD_MODE = True
    app_settings.CLOUD_EMBEDDING = EXPECTED_EMBEDDING
    app_settings.CLOUD_EMBEDDING_DIMENSION = int(manifest.get("embedding_dim") or 1536)


def _verify_persisted_runtime(manifest: dict[str, Any], org_id: str, cases_path: Path) -> dict[str, Any]:
    _configure_parent_for_verification(manifest)
    verification = verify_full_runtime(org_id, cases_path)
    if verification.get("strict_pass") is not True:
        raise RuntimeError("Independent persisted-runtime verification did not return strict_pass=true")
    if int(verification.get("persisted_literature_documents") or 0) != EXPECTED_DOCS:
        raise RuntimeError("Independent verification did not prove all TREC literature documents")
    alignment = verification.get("hnsw_vector_alignment") or {}
    if int(alignment.get("sample_count") or 0) <= 0 or float(alignment.get("minimum_cosine") or 0.0) < 0.9999:
        raise RuntimeError("Independent verification did not prove HNSW vector/label alignment")
    return verification


def _runtime_artifact_env(manifest: dict[str, Any]) -> dict[str, str]:
    """Return child-process paths bound to the verified publication artifacts."""
    hnsw = manifest.get("hnsw") or {}
    bm25 = manifest.get("bm25") or {}
    index_path = Path(str(hnsw.get("index_path") or "")).resolve()
    mapping_path = Path(str(hnsw.get("mapping_path") or "")).resolve()
    fts_path = Path(str(bm25.get("path") or "")).resolve()
    if not index_path.exists() or not mapping_path.exists() or not fts_path.exists():
        raise RuntimeError("Verified publication retrieval artifacts disappeared before matrix startup")
    artifact_dir = index_path.parent
    return {
        "HNSW_INDEX_PATH": str(index_path),
        "HNSW_MAPPING_PATH": str(mapping_path),
        "LEXICAL_FTS_PATH": str(fts_path),
        "FAISS_INDEX_PATH": str(artifact_dir / "faiss_unused.bin"),
        "FAISS_MAPPING_PATH": str(artifact_dir / "faiss_unused.json"),
        "TREE_INDEX_PATH": str(artifact_dir / "tree_unused.npy"),
        "TREE_MAPPING_PATH": str(artifact_dir / "tree_unused.json"),
    }


def _expected_model_revision() -> str:
    warmup = _load_json(MANIFEST_DIR / "warmup.json")
    model = warmup.get("model") or {}
    if model.get("model_id") != EXPECTED_LOCAL_MODEL or not model.get("complete"):
        raise RuntimeError("Warmup manifest does not certify the expected local MedSwin model")
    revision = str(model.get("revision") or "").strip()
    if not revision:
        raise RuntimeError("Warmup manifest does not pin a concrete MedSwin model revision")
    return revision


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


def _evidence_packet_stats(response: dict[str, Any]) -> tuple[int, int]:
    bundle = response.get("evidence_bundle") or {}
    passages = bundle.get("passages") or []
    return len(passages) if isinstance(passages, list) else 0, int(bundle.get("total_tokens") or 0)


def _strict_naive_errors(
    response: dict[str, Any],
    trace: dict[str, Any] | None,
    token_budget: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if response.get("pipeline") != "naive_rag":
        errors.append(f"unexpected_pipeline:{response.get('pipeline')}")
    if response.get("retrieval_backend") != "ann":
        errors.append(f"naive_not_ann:{response.get('retrieval_backend')}")
    if _degraded(response):
        errors.append("naive_degraded")
    if not str(response.get("answer") or "").strip():
        errors.append("empty_answer")
    passages, packet_tokens = _evidence_packet_stats(response)
    if passages <= 0:
        errors.append("naive_empty_evidence")
    if token_budget is not None:
        if packet_tokens <= 0:
            errors.append("naive_missing_evidence_token_count")
        if passages > 1 and packet_tokens > int(token_budget):
            errors.append(f"naive_evidence_budget_exceeded:{packet_tokens}>{token_budget}")
    if trace is None:
        errors.append("missing_trace")
        return sorted(set(errors))

    tools = trace.get("tool_calls") or []
    naive_tools = [item for item in tools if item.get("tool_name") == "retrieval.naive_dense"]
    if not naive_tools:
        errors.append("missing_naive_dense_trace")
    else:
        for item in naive_tools:
            params = item.get("parameters") or {}
            result = item.get("result") or {}
            if params.get("backend") != "ann" or int(result.get("count") or 0) <= 0:
                errors.append("naive_dense_trace_not_ann_or_empty")
                break

    if token_budget is not None:
        packing = [item for item in tools if item.get("tool_name") == "packing.naive"]
        if not packing:
            errors.append("missing_naive_packing_trace")
        else:
            latest = packing[-1]
            params = latest.get("parameters") or {}
            result = latest.get("result") or {}
            if int(params.get("token_budget") or 0) != int(token_budget):
                errors.append("naive_packing_budget_mismatch")
            if int(result.get("packed_passages") or 0) != passages:
                errors.append("naive_packing_passage_count_mismatch")
            if int(result.get("packed_tokens") or 0) != packet_tokens:
                errors.append("naive_packing_token_count_mismatch")

    if trace.get("rerank_traces"):
        errors.append("naive_used_reranker")
    if [item for item in tools if str(item.get("tool_name") or "").startswith("agent.")]:
        errors.append("naive_used_mac")
    if trace.get("sufficiency_checks"):
        errors.append("naive_used_sufficiency_gate")
    return sorted(set(errors))


def _strict_full_errors(
    response: dict[str, Any],
    trace: dict[str, Any] | None,
    token_budget: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if response.get("pipeline") != "medswin":
        errors.append(f"unexpected_pipeline:{response.get('pipeline')}")
    if _degraded(response):
        errors.append("medswin_degraded")
    if not str(response.get("answer") or "").strip():
        errors.append("empty_answer")
    passages, packet_tokens = _evidence_packet_stats(response)
    if passages <= 0:
        errors.append("full_empty_evidence")
    if token_budget is not None:
        if packet_tokens <= 0:
            errors.append("full_missing_evidence_token_count")
        if passages > 1 and packet_tokens > int(token_budget):
            errors.append(f"full_evidence_budget_exceeded:{packet_tokens}>{token_budget}")
    if not response.get("policy_decision"):
        errors.append("missing_policy_decision")
    if not response.get("sufficiency_decision"):
        errors.append("missing_sufficiency_decision")
    if not response.get("facet_matrix"):
        errors.append("missing_facet_matrix")
    if trace is None:
        errors.append("missing_trace")
        return sorted(set(errors))

    retrieval_traces = trace.get("retrieval_traces") or []
    rerank_traces = trace.get("rerank_traces") or []
    sufficiency_checks = trace.get("sufficiency_checks") or []
    tools = trace.get("tool_calls") or []
    if not retrieval_traces:
        errors.append("missing_hybrid_retrieval_trace")
    else:
        if not any(int(item.get("dense_count") or 0) > 0 for item in retrieval_traces):
            errors.append("dense_ann_stage_missing")
        if not any(int(item.get("lexical_count") or 0) > 0 for item in retrieval_traces):
            errors.append("bm25_stage_missing")
        if not any(int(item.get("union_count") or 0) > 0 for item in retrieval_traces):
            errors.append("hybrid_union_empty")

    hybrid_tools = [item for item in tools if item.get("tool_name") == "retrieval.hybrid"]
    if not hybrid_tools:
        errors.append("hybrid_tool_not_executed")
    elif not any(int((item.get("result") or {}).get("union_count") or 0) > 0 for item in hybrid_tools):
        errors.append("hybrid_tool_returned_empty_union")

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


def _retrieval_ranked_lit_doc_ids(
    pipeline: str,
    trace: dict[str, Any] | None,
    org_id: str,
) -> list[str]:
    """Resolve the pre-pack ranked literature list from trace chunk IDs."""
    if not trace:
        return []
    chunk_ids: list[str] = []
    if pipeline == "naive_rag":
        retrieval_traces = trace.get("retrieval_traces") or []
        if retrieval_traces:
            chunk_ids = [
                str(item.get("chunk_id") or "")
                for item in (retrieval_traces[-1].get("candidates") or [])
                if str(item.get("chunk_id") or "").strip()
            ]
    else:
        rerank_traces = trace.get("rerank_traces") or []
        if rerank_traces:
            chunk_ids = [
                str(item.get("chunk_id") or "")
                for item in (rerank_traces[-1].get("scores") or [])
                if str(item.get("chunk_id") or "").strip()
            ]
    if not chunk_ids:
        return []

    coll = get_sync_database()["chunks"]
    rows = coll.find(
        {"org_id": org_id, "chunk_id": {"$in": list(dict.fromkeys(chunk_ids))}},
        {"_id": 0, "chunk_id": 1, "doc_id": 1, "source_type": 1},
    )
    by_chunk = {str(row.get("chunk_id")): row for row in rows}
    ranked_docs: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        row = by_chunk.get(chunk_id) or {}
        if str(row.get("source_type") or "").upper() != "LIT":
            continue
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked_docs.append(doc_id)
    return ranked_docs


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


def _validate_local_health(health: dict[str, Any], expected_revision: str) -> None:
    if health.get("model") != EXPECTED_LOCAL_MODEL:
        raise RuntimeError(f"Local generator health reported unexpected model {health.get('model')}")
    if str(health.get("model_revision") or "").strip() != expected_revision:
        raise RuntimeError(
            f"Local generator revision {health.get('model_revision')!r} does not match warmup revision {expected_revision}"
        )
    if int(health.get("context_window") or 0) <= 0:
        raise RuntimeError("Local MedSwin server did not report a context window")
    if health.get("prompt_policy") != "fail_on_overflow_no_truncation":
        raise RuntimeError("Local MedSwin server is not enforcing fail-on-overflow/no-truncation")


def _start_local_medswin(expected_revision: str) -> tuple[subprocess.Popen[Any] | None, dict[str, Any]]:
    completions_url = os.getenv("MEDSWIN_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
    health_url = _health_url(completions_url)
    existing_health: dict[str, Any] | None = None
    try:
        existing_health = _wait_http(health_url, None, 2.0)
    except Exception:
        existing_health = None
    if existing_health is not None:
        _validate_local_health(existing_health, expected_revision)
        return None, existing_health

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
    try:
        health = _wait_http(
            health_url,
            process,
            float(os.getenv("MEDSWIN_LLM_STARTUP_TIMEOUT_S", "900")),
        )
        _validate_local_health(health, expected_revision)
        return process, health
    except Exception:
        _terminate(process)
        raise


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


def _expected_generator_model(generator: str) -> str:
    if generator == "medswin_local":
        return EXPECTED_LOCAL_MODEL
    return os.getenv("FOUNDRY_MODEL", "gpt-5.4")


def _api_env(generator: str, port: int, org_id: str, manifest: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    envelope = _shared_eval_envelope()
    env.update(_runtime_artifact_env(manifest))
    env.update(
        {
            "CLOUD_MODE": "true",
            "CLOUD_EMBEDDING": EXPECTED_EMBEDDING,
            "CLOUD_EMBEDDING_DIMENSION": str(int(manifest.get("embedding_dim") or 1536)),
            "CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE": "query",
            "CLOUD_RERANKER": EXPECTED_RERANKER,
            "GENERATION_BACKEND": generator,
            "FOUNDRY_MODEL": os.getenv("FOUNDRY_MODEL", "gpt-5.4"),
            "CLOUD_MODEL": os.getenv("FOUNDRY_MODEL", "gpt-5.4"),
            "APP_PORT": str(port),
            "BENCHMARK_ORG_ID": org_id,
            "MEDSWIN_BASE_URL": f"http://127.0.0.1:{port}",
            "NAIVE_ENABLE_MONGO_FALLBACK": "false",
            "ENABLE_BM25": "true",
            "DISABLE_DATASET_PRELOAD": "true",
            "TOKEN_BUDGET_B": str(envelope["token_budget"]),
            "AGENT_PASSAGE_LIMIT": str(envelope["agent_passage_limit"]),
            "AGENT_PASSAGE_MAX_CHARS": str(envelope["agent_passage_max_chars"]),
            "LLM_DEFAULT_MAX_TOKENS": str(envelope["generation_max_tokens"]),
        }
    )
    return env


def _validate_api_health(health: dict[str, Any], generator: str, manifest: dict[str, Any]) -> None:
    expected_model = _expected_generator_model(generator)
    failures: list[str] = []
    if health.get("status") != "healthy":
        failures.append(f"status={health.get('status')}")
    if health.get("cloud_mode") is not True:
        failures.append("cloud_mode_not_true")
    if health.get("generation_backend") != generator:
        failures.append(f"generation_backend={health.get('generation_backend')!r}")
    if health.get("generation_model") != expected_model:
        failures.append(f"generation_model={health.get('generation_model')!r}")
    if health.get("embedding_model") != EXPECTED_EMBEDDING:
        failures.append(f"embedding_model={health.get('embedding_model')!r}")
    if health.get("reranker_model") != EXPECTED_RERANKER:
        failures.append(f"reranker_model={health.get('reranker_model')!r}")
    if health.get("active_embedding_space") != f"cloud:{EXPECTED_EMBEDDING}":
        failures.append(f"embedding_space={health.get('active_embedding_space')!r}")
    if int(health.get("active_embedding_dimension") or 0) != int(manifest.get("embedding_dim") or 0):
        failures.append(f"embedding_dimension={health.get('active_embedding_dimension')!r}")
    if health.get("dataset_preload_disabled") is not True:
        failures.append("dataset_preload_not_disabled")
    if failures:
        raise RuntimeError("Full-eval API identity preflight failed: " + ", ".join(failures))


def _start_api(generator: str, port: int, org_id: str, manifest: dict[str, Any]) -> subprocess.Popen[Any]:
    log_path = REPO_ROOT / "logs" / f"full-eval-api-{generator}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO_ROOT,
        env=_api_env(generator, port, org_id, manifest),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        health = _wait_http(f"http://127.0.0.1:{port}/health", process, 180.0)
        _validate_api_health(health, generator, manifest)
        return process
    except Exception:
        _terminate(process)
        raise


async def _run_cell(
    cases: list[BenchmarkCase],
    pipeline: str,
    generator: str,
    port: int,
    org_id: str,
    gold_present: set[str],
    top_k: int,
    timeout_s: float,
    manifest: dict[str, Any],
) -> RunAudit:
    run_id = f"full-{pipeline}-{generator}-{uuid.uuid4().hex[:10]}"
    envelope = _shared_eval_envelope()
    run = RunAudit(
        run_id=run_id,
        dataset=EXPECTED_DATASET,
        config={
            "pipeline": pipeline,
            "generator": generator,
            "generator_model": _expected_generator_model(generator),
            "embedding_model": EXPECTED_EMBEDDING,
            "embedding_input_type": "query",
            "corpus_embedding_input_type": manifest.get("embedding_input_type"),
            "reranker_model": EXPECTED_RERANKER if pipeline == "medswin" else None,
            "cases": len(cases),
            "retrieval_top_k": top_k,
            "strict_full_corpus": True,
            "shared_generation_envelope": envelope,
            "runtime_artifacts": _runtime_artifact_env(manifest),
            "primary_retrieval_metrics": [
                "retrieval_ndcg_at_10",
                "retrieval_precision_at_10",
                "retrieval_recall_at_10",
                "retrieval_reciprocal_rank",
            ],
            "system_diagnostic": "msas",
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
            trace: dict[str, Any] | None = None
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
            else:
                trace_id = str(response.get("trace_id") or "")
                trace = _trace_document(trace_id, org_id) if trace_id else None
                strict_errors = (
                    _strict_naive_errors(response, trace, envelope["token_budget"])
                    if pipeline == "naive_rag"
                    else _strict_full_errors(response, trace, envelope["token_budget"])
                )

            retrieval_ranked_doc_ids = _retrieval_ranked_lit_doc_ids(pipeline, trace, org_id)
            if not retrieval_ranked_doc_ids:
                strict_errors.append("retrieval_literature_ranking_empty")

            case_audit = audit_case(
                case,
                response,
                _trace_summary(trace),
                errors=sorted(set(strict_errors)),
                available_doc_ids=gold_present,
                indexed_doc_ids=gold_present,
                pipeline=pipeline,
            )
            if retrieval_ranked_doc_ids:
                retrieval_metrics = ranked_trec_metrics(case, retrieval_ranked_doc_ids)
                case_audit.retrieval_ranked_doc_ids = retrieval_ranked_doc_ids
                case_audit.retrieval_ndcg_at_10 = retrieval_metrics["ndcg_at_10"]
                case_audit.retrieval_precision_at_10 = retrieval_metrics["precision_at_10"]
                case_audit.retrieval_recall_at_10 = retrieval_metrics["recall_at_10"]
                case_audit.retrieval_reciprocal_rank = retrieval_metrics["reciprocal_rank"]
            run.cases.append(case_audit)
            retrieval_ndcg = case_audit.retrieval_ndcg_at_10
            print(
                f"[full-matrix] {generator}/{pipeline} case {index}/{len(cases)} "
                f"errors={len(case_audit.errors)} "
                f"retrieval_ndcg@10={retrieval_ndcg if retrieval_ndcg is not None else 'NA'} "
                f"msas={case_audit.msas:.4f}",
                flush=True,
            )

    aggregate_run(run)
    retrieval_metric_cases = sum(1 for case in run.cases if case.retrieval_ndcg_at_10 is not None)
    run.diagnostics.update(
        {
            "strict_error_cases": sum(1 for case in run.cases if case.errors),
            "retrieval_metric_cases": retrieval_metric_cases,
            "strict_pass": (
                len(run.cases) == EXPECTED_QUERIES
                and retrieval_metric_cases == EXPECTED_QUERIES
                and all(not case.errors for case in run.cases)
            ),
            "generator": generator,
            "pipeline": pipeline,
            "expected_cases": EXPECTED_QUERIES,
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


def _delta_metrics(left: RunAudit, right: RunAudit) -> dict[str, float]:
    names = [
        "mean_retrieval_ndcg_at_10",
        "mean_retrieval_precision_at_10",
        "mean_retrieval_recall_at_10",
        "mean_retrieval_reciprocal_rank",
        "mean_final_evidence_ndcg_at_10",
        "mean_final_evidence_precision_at_10",
        "mean_final_evidence_recall_at_10",
        "mean_final_evidence_reciprocal_rank",
        "mean_evidence_doc_recall",
        "mean_citation_precision",
        "mean_groundedness_proxy",
        "mean_msas",
    ]
    return {name: _metric(left, name) - _metric(right, name) for name in names}


async def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    pipelines = resolve_pipelines(args.pipeline)
    manifest = _validate_runtime_manifest(args.org_id)
    cases_path = Path(args.cases_path or manifest.get("cases_path") or DEFAULT_CASES)
    cases = _load_cases(cases_path)

    verification = _verify_persisted_runtime(manifest, args.org_id, cases_path)
    gold_present = _qrel_presence(cases, args.org_id)
    expected_revision = _expected_model_revision()

    local_process, local_health = _start_local_medswin(expected_revision)
    results: dict[str, RunAudit] = {}
    try:
        for generator in ALL_GENERATORS:
            api_process = _start_api(generator, args.api_port, args.org_id, manifest)
            try:
                for pipeline in pipelines:
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
                        manifest,
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

            if generator == "medswin_local" and local_process is not None:
                _terminate(local_process)
                local_process = None
                await asyncio.sleep(1.0)
    finally:
        _terminate(local_process)

    expected_keys = expected_matrix_keys(pipelines)
    if set(results) != expected_keys:
        raise RuntimeError(f"Evaluation matrix is incomplete: got={sorted(results)} expected={sorted(expected_keys)}")

    matrix = {
        "created_at": _now(),
        "selected_pipelines": list(pipelines),
        "publication_complete": is_publication_matrix(pipelines),
        "complete_trec_runtime": manifest,
        "persisted_runtime_verification": verification,
        "local_medswin_health": local_health,
        "local_medswin_revision": expected_revision,
        "shared_generation_envelope": _shared_eval_envelope(),
        "retrieval_top_k": args.top_k,
        "primary_retrieval_metrics": [
            "mean_retrieval_ndcg_at_10",
            "mean_retrieval_precision_at_10",
            "mean_retrieval_recall_at_10",
            "mean_retrieval_reciprocal_rank",
        ],
        "secondary_rag_system_metrics": [
            "mean_final_evidence_ndcg_at_10",
            "mean_evidence_doc_recall",
            "mean_citation_precision",
            "mean_groundedness_proxy",
            "mean_msas",
        ],
        "runtime_artifacts": _runtime_artifact_env(manifest),
        "cells": {
            key: {
                "run_id": run.run_id,
                "strict_pass": bool(run.diagnostics.get("strict_pass")),
                "aggregate": run.aggregate,
                "error_cases": int(run.diagnostics.get("strict_error_cases") or 0),
                "retrieval_metric_cases": int(run.diagnostics.get("retrieval_metric_cases") or 0),
                "cases": len(run.cases),
            }
            for key, run in results.items()
        },
        "comparisons": {},
        "metric_notes": {
            "retrieval": (
                "Pre-pack literature rankings are scored at cutoff 10 from TREC qrels. "
                "These are track-aligned diagnostics, not official NIST inferred/sample-eval leaderboard scores."
            ),
            "final_evidence": "Post-pack evidence packet shown to the generator under the shared token budget.",
            "msas": "Architecture/system-audit diagnostic; not the sole model-comparison metric.",
        },
    }
    for generator in ALL_GENERATORS:
        naive_key = f"naive_rag:{generator}"
        full_key = f"medswin:{generator}"
        if naive_key in results and full_key in results:
            matrix["comparisons"][f"full_minus_naive:{generator}"] = _delta_metrics(
                results[full_key], results[naive_key]
            )
    for pipeline in pipelines:
        local = results[f"{pipeline}:medswin_local"]
        cloud = results[f"{pipeline}:foundry"]
        matrix["comparisons"][f"gpt54_minus_medswin7b:{pipeline}"] = _delta_metrics(cloud, local)

    matrix["strict_pass"] = (
        verification.get("strict_pass") is True
        and all(
            cell["strict_pass"]
            and cell["cases"] == EXPECTED_QUERIES
            and cell["retrieval_metric_cases"] == EXPECTED_QUERIES
            for cell in matrix["cells"].values()
        )
    )
    path = _run_output_dir() / f"full-matrix-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
    matrix["matrix_path"] = str(path)
    if not matrix["strict_pass"]:
        raise RuntimeError(f"One or more full evaluation cells failed strict architecture validation; matrix={path}")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default=os.getenv("BENCHMARK_ORG_ID", "bench-org"))
    parser.add_argument(
        "--pipeline",
        choices=("medswin", "naive_rag", "both"),
        default="both",
        help="Which retrieval systems to run. Default both is the publication 2x2. "
        "A subset is architecture-strict for the selected cells but not publication-complete.",
    )
    parser.add_argument("--cases-path", default="")
    parser.add_argument("--api-port", type=int, default=int(os.getenv("FULL_EVAL_API_PORT", "8110")))
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("FULL_EVAL_TOP_K", str(DEFAULT_RETRIEVAL_TOP_K))),
        help="Naive pre-pack dense retrieval depth; default 10 so P@10/nDCG@10 are observable before context packing.",
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("FULL_EVAL_REQUEST_TIMEOUT_S", "600")))
    args = parser.parse_args()
    if args.top_k < 10:
        parser.error("--top-k must be >=10 for the strict publication matrix")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    matrix = asyncio.run(run_matrix(parse_args()))
    print(
        json.dumps(
            {
                "strict_pass": matrix["strict_pass"],
                "publication_complete": matrix["publication_complete"],
                "selected_pipelines": matrix["selected_pipelines"],
                "matrix_path": matrix["matrix_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
