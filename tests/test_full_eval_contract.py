import sqlite3

import pytest

from app.core.indexing.hnsw import SQLiteLabelMapping
from eval.scripts.run_full_matrix import (
    EXPECTED_LOCAL_MODEL,
    _runtime_artifact_env,
    _strict_full_errors,
    _strict_naive_errors,
    _validate_local_health,
)


def _full_response():
    return {
        "pipeline": "medswin",
        "answer": "Grounded answer",
        "evidence_bundle": {"passages": [{"chunk_id": "c1", "doc_id": "d1"}]},
        "policy_decision": {"passed": True},
        "sufficiency_decision": {"passed": True},
        "facet_matrix": {"rows": []},
        "degraded_mode": {},
    }


def _full_trace():
    agent_names = ["emr", "guideline", "safety", "quality", "critic"]
    return {
        "retrieval_traces": [
            {
                "dense_count": 12,
                "lexical_count": 10,
                "union_count": 18,
                "candidates": [
                    {"chunk_id": "c1", "retrieved_by": ["dense"]},
                    {"chunk_id": "c2", "retrieved_by": ["lexical_bm25_fts5"]},
                ],
            }
        ],
        "rerank_traces": [
            {
                "calibration_version": "identity:cohere-v2",
                "scores": [{"chunk_id": "c1", "p_hat": 0.9}],
            }
        ],
        "sufficiency_checks": [{"passed": True}],
        "tool_calls": [
            {"tool_name": "retrieval.hybrid", "result": {"union_count": 18}},
            {
                "tool_name": "retrieval.rerank",
                "result": {"calibration_version": "identity:cohere-v2", "n": 18},
            },
            *[
                {
                    "tool_name": f"agent.{name}",
                    "result": {"claims": 1, "degraded": False, "error": None},
                }
                for name in agent_names
            ],
        ],
        "messages": [{"agent_id": name, "content": "ok"} for name in agent_names],
        "evidence_bundle": {"passages": [{"chunk_id": "c1"}]},
    }


def test_full_contract_accepts_complete_two_stage_mac_trace():
    assert _strict_full_errors(_full_response(), _full_trace()) == []


def test_full_contract_rejects_bm25_only_when_ann_stage_is_missing():
    trace = _full_trace()
    trace["retrieval_traces"][0]["dense_count"] = 0

    errors = _strict_full_errors(_full_response(), trace)

    assert "dense_ann_stage_missing" in errors


def test_full_contract_rejects_ann_only_when_bm25_stage_is_missing():
    trace = _full_trace()
    trace["retrieval_traces"][0]["lexical_count"] = 0

    errors = _strict_full_errors(_full_response(), trace)

    assert "bm25_stage_missing" in errors


def test_full_contract_rejects_reranker_fail_open_marker():
    trace = _full_trace()
    trace["rerank_traces"][0]["calibration_version"] = "identity:rerank-error"

    errors = _strict_full_errors(_full_response(), trace)

    assert "reranker_failed_open" in errors


def test_full_contract_rejects_missing_mac_specialist():
    trace = _full_trace()
    trace["tool_calls"] = [
        item for item in trace["tool_calls"] if item.get("tool_name") != "agent.critic"
    ]

    errors = _strict_full_errors(_full_response(), trace)

    assert "mac_agents_missing:critic" in errors


def test_naive_contract_requires_ann_and_forbids_full_system_artifacts():
    response = {
        "pipeline": "naive_rag",
        "retrieval_backend": "ann",
        "answer": "answer",
        "evidence_bundle": {"passages": [{"chunk_id": "c1"}]},
        "degraded_mode": {},
    }
    trace = {
        "tool_calls": [
            {
                "tool_name": "retrieval.naive_dense",
                "parameters": {"backend": "ann", "top_k": 5},
                "result": {"count": 5},
            }
        ],
        "rerank_traces": [],
        "sufficiency_checks": [],
    }

    assert _strict_naive_errors(response, trace) == []

    response["retrieval_backend"] = "mongo_cosine"
    assert any(error.startswith("naive_not_ann") for error in _strict_naive_errors(response, trace))


def test_runtime_artifact_binding_ignores_generic_dev_paths(tmp_path, monkeypatch):
    index = tmp_path / "publication" / "hnsw_index.bin"
    mapping = tmp_path / "publication" / "hnsw_mapping.sqlite"
    bm25 = tmp_path / "publication" / "bm25.sqlite"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"index")
    mapping.write_bytes(b"mapping")
    bm25.write_bytes(b"bm25")

    monkeypatch.setenv("HNSW_INDEX_PATH", "/tmp/dev-index.bin")
    monkeypatch.setenv("FAISS_INDEX_PATH", "/tmp/stale-dev-faiss.bin")
    manifest = {
        "hnsw": {"index_path": str(index), "mapping_path": str(mapping)},
        "bm25": {"path": str(bm25)},
    }

    bound = _runtime_artifact_env(manifest)

    assert bound["HNSW_INDEX_PATH"] == str(index.resolve())
    assert bound["HNSW_MAPPING_PATH"] == str(mapping.resolve())
    assert bound["LEXICAL_FTS_PATH"] == str(bm25.resolve())
    assert bound["FAISS_INDEX_PATH"] == str(index.parent / "faiss_unused.bin")
    assert bound["FAISS_INDEX_PATH"] != "/tmp/stale-dev-faiss.bin"


def test_local_generator_health_requires_exact_pinned_revision():
    health = {
        "model": EXPECTED_LOCAL_MODEL,
        "model_revision": "abc123",
        "context_window": 4096,
        "prompt_policy": "fail_on_overflow_no_truncation",
    }
    _validate_local_health(health, "abc123")

    with pytest.raises(RuntimeError, match="does not match warmup revision"):
        _validate_local_health(health, "different")


def test_sqlite_hnsw_label_mapping_is_lazy_and_exact(tmp_path):
    path = tmp_path / "mapping.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mapping(label INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE)")
    conn.executemany(
        "INSERT INTO mapping(label, chunk_id) VALUES (?, ?)",
        [(0, "chunk-a"), (1, "chunk-b")],
    )
    conn.commit()
    conn.close()

    mapping = SQLiteLabelMapping(str(path))
    try:
        assert mapping.get("0") == "chunk-a"
        assert mapping.get("1") == "chunk-b"
        assert mapping.get("2") is None
        assert mapping.get("not-a-label") is None
    finally:
        mapping.close()
