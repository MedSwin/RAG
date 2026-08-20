"""Hybrid ANN: merge HNSW and IVF-PQ / FAISS candidates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class HybridIndex:
    """Query HNSW and FAISS IVF, reloading completed artifact generations."""

    def __init__(self):
        self._hnsw = None
        self._faiss = None
        self._embedding_dim: Optional[int] = None
        self._artifact_signature: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def _path_signature(path_value: str | Path) -> Tuple[str, int, int]:
        path = Path(path_value)
        try:
            stat = path.stat()
            return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return str(path.resolve()), 0, 0

    def _generation_signature(self, index_path: str, mapping_path: str) -> Tuple[Any, ...]:
        """Use manifest commit point when present, otherwise raw file signatures.

        Both ordinary and publication builders write the sidecar manifest only
        after index+mapping construction. While a background rebuild is
        replacing those files, a live process therefore continues using its old
        in-memory generation until the manifest changes. This prevents loading a
        new index with an old mapping during the replacement window.
        """
        manifest = Path(index_path).with_name(f"{Path(index_path).name}.manifest.json")
        if manifest.exists():
            return ("manifest", self._path_signature(manifest))
        return (
            "files",
            self._path_signature(index_path),
            self._path_signature(mapping_path),
        )

    def _current_signature(self) -> Tuple[Any, ...]:
        return (
            self._generation_signature(settings.HNSW_INDEX_PATH, settings.HNSW_MAPPING_PATH),
            self._generation_signature(settings.FAISS_INDEX_PATH, settings.FAISS_MAPPING_PATH),
        )

    @staticmethod
    def _close_mapping(index: Any) -> None:
        mapping = getattr(index, "mapping", None)
        close = getattr(mapping, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def _load(self, embedding_dim: int) -> None:
        signature = self._current_signature()
        if (
            self._hnsw is not None
            and self._embedding_dim == embedding_dim
            and self._artifact_signature == signature
        ):
            return

        old_hnsw = self._hnsw
        old_faiss = self._faiss
        new_hnsw = None
        new_faiss = None
        try:
            from app.core.indexing import load_hnsw_index

            new_hnsw = load_hnsw_index(
                embedding_dim,
                settings.HNSW_INDEX_PATH,
                settings.HNSW_MAPPING_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HNSW load failed: %s", exc)
        try:
            from app.core.indexing import load_faiss_ivf_index

            new_faiss = load_faiss_ivf_index(
                embedding_dim,
                settings.FAISS_INDEX_PATH,
                settings.FAISS_MAPPING_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("FAISS load skipped/failed: %s", exc)

        if new_hnsw is None and new_faiss is None and (old_hnsw is not None or old_faiss is not None):
            logger.warning("New ANN generation could not be loaded; retaining previous in-memory generation")
            return

        self._hnsw = new_hnsw
        self._faiss = new_faiss
        self._embedding_dim = embedding_dim
        self._artifact_signature = signature
        if old_hnsw is not None and old_hnsw is not new_hnsw:
            self._close_mapping(old_hnsw)
        if old_faiss is not None and old_faiss is not new_faiss:
            self._close_mapping(old_faiss)

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

    def close(self) -> None:
        self._close_mapping(self._hnsw)
        self._close_mapping(self._faiss)
        self._hnsw = None
        self._faiss = None
