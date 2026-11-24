import asyncio
from pathlib import Path

from app.retrieval import lexical


def test_fts_path_blocks_org_id_path_traversal(monkeypatch, tmp_path):
    monkeypatch.delenv("LEXICAL_FTS_PATH", raising=False)
    monkeypatch.setattr(lexical, "CACHE_DIR", tmp_path)

    assert lexical.LexicalRetriever._fts_path("../outside") is None
    assert lexical.LexicalRetriever._fts_path("org/sub") is None
    assert lexical.LexicalRetriever._fts_path("..") is None

    valid = lexical.LexicalRetriever._fts_path("demo-org_1")
    assert valid == (tmp_path / "demo-org_1_fts.sqlite").resolve()


def test_bm25_bundle_stays_in_memory_and_writes_no_pickle_files(monkeypatch, tmp_path):
    monkeypatch.setattr(lexical, "CACHE_DIR", tmp_path)

    class _Cursor:
        async def to_list(self, length):
            return [{"chunk_id": "c1", "text": "Patient chart beta blocker allergy."}]

    class _Chunks:
        async def count_documents(self, query, limit=None):
            return 1

        def find(self, query, projection=None):
            return _Cursor()

    class _DB:
        chunks = _Chunks()

    monkeypatch.setattr(lexical, "get_database", lambda: _DB())
    retriever = lexical.LexicalRetriever()
    bundle = asyncio.run(
        retriever._get_bundle("org-a", None, "patient-1", {})
    )
    assert bundle is not None
    assert bundle["chunk_ids"] == ["c1"]
    assert list(tmp_path.glob("*.pkl")) == []
    assert "pickle" not in Path(lexical.__file__).read_text(encoding="utf-8")
