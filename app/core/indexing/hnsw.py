"""HNSW index builder and loader with scalable label mappings."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from app.core.config import settings
from app.core.indexing.base import BaseIndexBuilder

logger = logging.getLogger(__name__)


class SQLiteLabelMapping:
    """Lazy label→chunk lookup for full-corpus indexes.

    A JSON dict for millions of chunk IDs can consume substantial RAM before a
    single query is served. Full TREC builds therefore store the mapping in
    SQLite and resolve only the HNSW labels returned by each query. Existing
    JSON mappings remain fully supported.
    """

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)

    def get(self, key: str, default=None):
        try:
            label = int(key)
        except (TypeError, ValueError):
            return default
        row = self._conn.execute("SELECT chunk_id FROM mapping WHERE label = ?", (label,)).fetchone()
        return row[0] if row else default

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _sqlite_mapping(path: str) -> bool:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size < 16:
        return False
    try:
        with file_path.open("rb") as handle:
            return handle.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


class HNSWIndexBuilder(BaseIndexBuilder):
    """Builder for HNSW (Hierarchical Navigable Small World) indexes."""

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = {
                "M": 16,
                "ef_construction": 200,
                "space": "cosine",
                "max_elements": 100000,
            }
        super().__init__(embedding_dim, config)
        self.index = None
        self.mapping: Any = {}

    def build(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        index_path: str,
        mapping_path: str,
    ) -> Dict[str, Any]:
        """Build an in-memory HNSW index for normal/smaller corpora."""
        try:
            valid_embeddings, valid_chunk_ids = self._validate_embeddings(embeddings, chunk_ids)
            if not valid_embeddings:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "total_vectors": 0,
                    "message": "No valid embeddings found",
                }

            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            self.index = hnswlib.Index(space=self.config.get("space", "cosine"), dim=self.embedding_dim)

            dataset_size = len(valid_embeddings)
            adaptive_m = self.config.get("M") or (16 if dataset_size < 50000 else 32)
            adaptive_m = min(max(int(adaptive_m), 8), max(dataset_size - 1, 8))
            adaptive_ef_construction = max(int(self.config.get("ef_construction", 200)), adaptive_m * 12)
            self.config.update(
                {
                    "M": adaptive_m,
                    "ef_construction": adaptive_ef_construction,
                    "max_elements": max(int(self.config.get("max_elements", 0)), dataset_size),
                }
            )

            self.index.init_index(
                max_elements=self.config["max_elements"],
                ef_construction=self.config["ef_construction"],
                M=self.config["M"],
            )
            self.index.add_items(embeddings_array)
            self.mapping = {str(i): chunk_id for i, chunk_id in enumerate(valid_chunk_ids)}

            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            self.index.save_index(index_path)
            self._save_mapping(mapping_path, self.mapping)
            logger.info("HNSW index built successfully with %s vectors", len(valid_embeddings))
            return {
                "success": True,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": len(valid_embeddings),
                "message": f"Index built successfully with {len(valid_embeddings)} vectors",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error building HNSW index: %s", exc)
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": 0,
                "message": f"Index building failed: {exc}",
            }

    def load(self, index_path: str, mapping_path: str) -> bool:
        """Load existing HNSW and either JSON or SQLite label mapping."""
        try:
            if isinstance(self.mapping, SQLiteLabelMapping):
                self.mapping.close()
            self.mapping = (
                SQLiteLabelMapping(mapping_path)
                if _sqlite_mapping(mapping_path)
                else self._load_mapping(mapping_path)
            )
            self.index = hnswlib.Index(space=self.config.get("space", "cosine"), dim=self.embedding_dim)
            self.index.load_index(str(index_path))
            logger.info(
                "HNSW index loaded from %s with %s mapping",
                index_path,
                "SQLite" if isinstance(self.mapping, SQLiteLabelMapping) else "JSON",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading HNSW index: %s", exc)
            return False

    def query(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not loaded or built")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        available = int(getattr(self.index, "element_count", 0))
        if available <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        requested_k = max(1, min(int(top_k), available))

        # Search breadth may need to grow after the hnswlib contiguous-array/
        # ef-too-small errors, but it must never scale with the entire index.
        # On the complete TREC runtime ``available`` is in the millions; using
        # it as the retry ceiling can turn one difficult query into an
        # unbounded-memory/latency event. HNSW_MAX_EF_SEARCH is the operational
        # ceiling, except that a caller asking for a larger k must still be
        # allowed enough ef to satisfy that explicit request.
        ef_search = max(requested_k * 2, 50)
        max_ef = max(int(settings.HNSW_MAX_EF_SEARCH), ef_search)
        last_error = None
        while ef_search <= max_ef:
            try:
                self.index.set_ef(ef_search)
                labels, distances = self.index.knn_query(query_embedding, k=requested_k)
                return labels[0], distances[0]
            except RuntimeError as exc:
                last_error = exc
                message = str(exc).lower()
                if "contiguous 2d array" not in message and "ef or m is too small" not in message:
                    raise
                if ef_search >= max_ef:
                    break
                ef_search = min(ef_search * 2, max_ef)
        raise last_error or RuntimeError("HNSW query failed")

    def get_index_info(self) -> Dict[str, Any]:
        if self.index is None:
            return {"type": "hnsw", "loaded": False, "message": "Index not loaded"}
        return {
            "type": "hnsw",
            "loaded": True,
            "dimension": self.embedding_dim,
            "total_vectors": self.index.element_count,
            "space": self.config.get("space", "cosine"),
            "M": self.config.get("M", 16),
            "ef_construction": self.config.get("ef_construction", 200),
            "mapping_backend": "sqlite" if isinstance(self.mapping, SQLiteLabelMapping) else "json",
        }


def load_hnsw_index(embedding_dim: int, index_path: str, mapping_path: str) -> HNSWIndexBuilder:
    builder = HNSWIndexBuilder(embedding_dim)
    if not builder.load(index_path, mapping_path):
        raise RuntimeError(f"Failed to load HNSW index from {index_path}")
    return builder