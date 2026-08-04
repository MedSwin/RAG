"""Hybrid ANN: merge HNSW and IVF-PQ / FAISS candidates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class HybridIndex:
    """Query HNSW and FAISS IVF, merge by dense similarity."""

    def __init__(self):
        self._hnsw = None
        self._faiss = None
        self._embedding_dim: Optional[int] = None

    def _load(self, embedding_dim: int) -> None:
        if self._hnsw is not None and self._embedding_dim == embedding_dim:
            return
        self._embedding_dim = embedding_dim
        try:
            from app.core.indexing import load_hnsw_index

            self._hnsw = load_hnsw_index(
                embedding_dim,
                settings.HNSW_INDEX_PATH,
                settings.HNSW_MAPPING_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HNSW load failed: %s", exc)
            self._hnsw = None
        try:
            from app.core.indexing import load_faiss_ivf_index

            self._faiss = load_faiss_ivf_index(
                embedding_dim,
                settings.FAISS_INDEX_PATH,
                settings.FAISS_MAPPING_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("FAISS load skipped/failed: %s", exc)
            self._faiss = None

    def query(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Return chunk_ids and dense similarity scores."""
        embedding_dim = int(query_embedding.shape[-1])
        self._load(embedding_dim)
        scores: Dict[str, float] = {}

        def _ingest(index: Any, label: str) -> None:
            if index is None:
                return
            try:
                labels, distances = index.query(query_embedding.reshape(1, -1), k)
                labels = np.asarray(labels).reshape(-1)
                distances = np.asarray(distances).reshape(-1)
                for i, lab in enumerate(labels):
                    chunk_id = index.mapping.get(str(int(lab)))
                    if not chunk_id:
                        continue
                    sim = 1.0 - float(distances[i]) if i < len(distances) else 0.0
                    prev = scores.get(chunk_id)
                    if prev is None or sim > prev:
                        scores[chunk_id] = sim
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s query failed: %s", label, exc)

        _ingest(self._hnsw, "HNSW")
        _ingest(self._faiss, "FAISS")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [chunk_id for chunk_id, _ in ranked], {cid: scores[cid] for cid, _ in ranked}
