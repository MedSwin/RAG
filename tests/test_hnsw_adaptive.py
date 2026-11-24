import numpy as np
import pytest

from app.core.indexing.hnsw import HNSWIndexBuilder


class _FlakyHnswIndex:
    element_count = 3

    def __init__(self):
        self.efs = []
        self.calls = 0

    def set_ef(self, value):
        self.efs.append(value)

    def knn_query(self, query_embedding, k):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
        return np.array([[0, 1, 2]], dtype=np.int64), np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


class _AlwaysFailLargeHnswIndex:
    element_count = 1_500_000

    def __init__(self):
        self.efs = []

    def set_ef(self, value):
        self.efs.append(value)

    def knn_query(self, query_embedding, k):
        raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")


def test_hnsw_query_clamps_k_and_increases_ef_until_success():
    builder = HNSWIndexBuilder(embedding_dim=2)
    fake = _FlakyHnswIndex()
    builder.index = fake

    labels, distances = builder.query(np.array([1.0, 0.0], dtype=np.float32), top_k=80)

    assert labels.tolist() == [0, 1, 2]
    assert distances.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert fake.efs[0] == 50
    assert fake.efs[1] > fake.efs[0]


def test_hnsw_retry_ceiling_does_not_scale_to_full_index(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "HNSW_MAX_EF_SEARCH", 2000)
    builder = HNSWIndexBuilder(embedding_dim=2)
    fake = _AlwaysFailLargeHnswIndex()
    builder.index = fake

    with pytest.raises(RuntimeError, match="contiguous 2D array"):
        builder.query(np.array([1.0, 0.0], dtype=np.float32), top_k=80)

    assert fake.efs
    assert fake.efs[-1] == 2000
    assert max(fake.efs) == 2000
    assert len(fake.efs) < 10


def test_index_build_message_hides_the_exception_text(monkeypatch):
    from app.cli.surfaces import probe
    from app.core.indexing.faiss import FAISSIndexBuilder
    from app.core.indexing.tree import TreeIndexBuilder

    def _boom(*_args, **_kwargs):
        raise RuntimeError("index path 555-234-5678")

    for builder in (HNSWIndexBuilder(2), FAISSIndexBuilder(2), TreeIndexBuilder(2)):
        builder._validate_embeddings = _boom
        result = builder.build([[1.0, 0.0]], ["c1"], "/tmp/index", "/tmp/map")
        assert result["success"] is False
        assert result["message"] == "Index building failed: RuntimeError"
        assert "555-234-5678" not in result["message"]

    def _http(*_args, **_kwargs):
        raise RuntimeError("connection 555-234-5678")

    monkeypatch.setattr("app.cli.surfaces.http_json", _http)
    status = probe("http://127.0.0.1:9/health")
    assert status["error"] == "RuntimeError"
    assert "555-234-5678" not in str(status)