import sqlite3

import pytest

from app.core.indexing.hnsw import SQLiteLabelMapping
from app.medswin.naive import _truncate_context
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from benchmarks.trec_cds2016.contract import (
    EXPECTED_CHUNKING_CONTRACT,
    EXPECTED_DATASET,
    EXPECTED_DOCUMENTS,
    PREPARATION_CONTRACT_VERSION,
    chunker_sha256,
)
from benchmarks.trec_cds2016.runtime import EXPECTED_LOCAL_MODEL, runtime_artifact_env, validate_local_health
from benchmarks.trec_cds2016.prepare.runtime import (
    EXPECTED_DATASET as BUILDER_DATASET,
    EXPECTED_DOCS as BUILDER_DOCUMENTS,
    PREPARATION_CONTRACT_VERSION as BUILDER_PREPARATION_VERSION,
    _chunker_sha256,
)
from benchmarks.trec_cds2016.prepare.verify import _fingerprint_ids


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

    bound = runtime_artifact_env(manifest)

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
    validate_local_health(health, "abc123")

    with pytest.raises(RuntimeError, match="does not match warmup revision"):
        validate_local_health(health, "different")


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
    assert EXPECTED_DATASET == BUILDER_DATASET
    assert EXPECTED_DOCUMENTS == BUILDER_DOCUMENTS
    assert PREPARATION_CONTRACT_VERSION == BUILDER_PREPARATION_VERSION
    assert EXPECTED_CHUNKING_CONTRACT == "app.medswin.chunking.section_chunks/full-body"
    assert chunker_sha256() == _chunker_sha256()
