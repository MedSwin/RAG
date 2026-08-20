import sqlite3

import pytest

from app.core.indexing.hnsw import SQLiteLabelMapping
from app.medswin.naive import _truncate_context
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from eval.app.audit import ranked_trec_metrics
from eval.app.full_contract import (
    EXPECTED_CHUNKING_CONTRACT as SHARED_CHUNKING_CONTRACT,
    EXPECTED_DATASET as SHARED_DATASET,
    EXPECTED_DOCUMENTS as SHARED_DOCUMENTS,
    PREPARATION_CONTRACT_VERSION as SHARED_PREPARATION_VERSION,
    chunker_sha256,
)
from eval.app.schemas import BenchmarkCase
from eval.scripts.prepare_full_trec_runtime import (
    EXPECTED_DATASET as BUILDER_DATASET,
    EXPECTED_DOCS as BUILDER_DOCUMENTS,
    PREPARATION_CONTRACT_VERSION as BUILDER_PREPARATION_VERSION,
    _chunker_sha256,
)
from eval.scripts.run_full_matrix import (
    EXPECTED_LOCAL_MODEL,
    _runtime_artifact_env,
    _strict_full_errors,
    _strict_naive_errors,
    _validate_local_health,
)
from eval.scripts.verify_full_trec_runtime import _fingerprint_ids


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
    assert "dense_ann_stage_missing" in _strict_full_errors(_full_response(), trace)


def test_full_contract_rejects_ann_only_when_bm25_stage_is_missing():
    trace = _full_trace()
    trace["retrieval_traces"][0]["lexical_count"] = 0
    assert "bm25_stage_missing" in _strict_full_errors(_full_response(), trace)


def test_full_contract_rejects_reranker_fail_open_marker():
    trace = _full_trace()
    trace["rerank_traces"][0]["calibration_version"] = "identity:rerank-error"
    assert "reranker_failed_open" in _strict_full_errors(_full_response(), trace)


def test_full_contract_rejects_missing_mac_specialist():
    trace = _full_trace()
    trace["tool_calls"] = [
        item for item in trace["tool_calls"] if item.get("tool_name") != "agent.critic"
    ]
    assert "mac_agents_missing:critic" in _strict_full_errors(_full_response(), trace)


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


def test_cross_store_fingerprint_is_order_independent_but_identity_sensitive():
    forward = _fingerprint_ids(["chunk-a", "chunk-b", "chunk-c"])
    reverse = _fingerprint_ids(["chunk-c", "chunk-b", "chunk-a"])
    substituted = _fingerprint_ids(["chunk-a", "chunk-b", "chunk-x"])
    duplicated = _fingerprint_ids(["chunk-a", "chunk-b", "chunk-b"])

    assert forward == reverse
    assert forward != substituted
    assert forward != duplicated
    assert forward["count"] == 3


def test_trec_ranked_metrics_use_graded_qrels_and_fixed_p10_denominator():
    case = BenchmarkCase(
        case_id="1",
        query="clinical question",
        gold_doc_ids=["d1", "d2"],
        metadata={"relevance_grades": {"d1": 2, "d2": 1}},
    )

    ideal = ranked_trec_metrics(case, ["d1", "d2"])
    reversed_ranking = ranked_trec_metrics(case, ["d2", "d1"])

    assert ideal["ndcg_at_10"] == pytest.approx(1.0)
    assert ideal["precision_at_10"] == pytest.approx(0.2)
    assert ideal["recall_at_10"] == pytest.approx(1.0)
    assert ideal["reciprocal_rank"] == pytest.approx(1.0)
    assert reversed_ranking["ndcg_at_10"] < ideal["ndcg_at_10"]


def test_naive_context_packer_uses_token_budget_before_character_ceiling():
    passages = [
        CandidatePassage(
            chunk_id="c1",
            doc_id="d1",
            source_type=SourceType.LIT,
            text="one two three four",
        ),
        CandidatePassage(
            chunk_id="c2",
            doc_id="d2",
            source_type=SourceType.LIT,
            text="five six seven eight",
        ),
    ]

    packed = _truncate_context(passages, max_chars=10000, token_budget=5)

    assert [passage.chunk_id for passage in packed] == ["c1"]


def test_resume_preflight_contract_matches_full_corpus_builder():
    # The shared lightweight contract is deliberately dependency-light. The
    # builder still owns its historical constants, so this test prevents those
    # two identities from drifting without requiring the builder to re-export
    # every shared constant.
    assert SHARED_DATASET == BUILDER_DATASET
    assert SHARED_DOCUMENTS == BUILDER_DOCUMENTS
    assert SHARED_PREPARATION_VERSION == BUILDER_PREPARATION_VERSION
    assert SHARED_CHUNKING_CONTRACT == "app.medswin.chunking.section_chunks/full-body"
    assert chunker_sha256() == _chunker_sha256()
