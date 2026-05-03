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


def test_hnsw_query_clamps_k_and_increases_ef_until_success():
    builder = HNSWIndexBuilder(embedding_dim=2)
    fake = _FlakyHnswIndex()
    builder.index = fake

    labels, distances = builder.query(np.array([1.0, 0.0], dtype=np.float32), top_k=80)

    assert labels.tolist() == [0, 1, 2]
    assert distances.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert fake.efs[0] == 50
    assert fake.efs[1] > fake.efs[0]
