"""HNSW index builder and loader."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import hnswlib
from pathlib import Path

from app.core.indexing.base import BaseIndexBuilder
from app.core.config import settings

logger = logging.getLogger(__name__)


class HNSWIndexBuilder(BaseIndexBuilder):
    """Builder for HNSW (Hierarchical Navigable Small World) indexes."""
    
    def __init__(
        self,
        embedding_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HNSW index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
            config: Configuration dict with M, ef_construction, space, max_elements
        """
        if config is None:
            config = {
                "M": 16,
                "ef_construction": 200,
                "space": "cosine",
                "max_elements": 100000
            }
        
        super().__init__(embedding_dim, config)
        self.index = None
        self.mapping = {}
    
    def build(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        index_path: str,
        mapping_path: str
    ) -> Dict[str, Any]:
        """
        Build HNSW index from embeddings.
        
        Args:
            embeddings: List of embeddings
            chunk_ids: List of chunk IDs corresponding to embeddings
            index_path: Path to save index file
            mapping_path: Path to save mapping file
            
        Returns:
            Dict with success status and metadata
        """
        try:
            # Validate embeddings
            valid_embeddings, valid_chunk_ids = self._validate_embeddings(
                embeddings, chunk_ids
            )
            
            if not valid_embeddings:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "total_vectors": 0,
                    "message": "No valid embeddings found"
                }
            
            # Convert to numpy array
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            
            # Create HNSW index
            self.index = hnswlib.Index(
                space=self.config.get("space", "cosine"),
                dim=self.embedding_dim
            )
            
            dataset_size = len(valid_embeddings)
            adaptive_m = self.config.get("M") or (16 if dataset_size < 50000 else 32)
            adaptive_m = min(max(int(adaptive_m), 8), max(dataset_size - 1, 8))
            adaptive_ef_construction = max(
                int(self.config.get("ef_construction", 200)),
                adaptive_m * 12,
            )
            self.config.update({
                "M": adaptive_m,
                "ef_construction": adaptive_ef_construction,
                "max_elements": max(int(self.config.get("max_elements", 0)), dataset_size),
            })

            self.index.init_index(
                max_elements=self.config["max_elements"],
                ef_construction=self.config["ef_construction"],
                M=self.config["M"]
            )
            
            self.index.add_items(embeddings_array)
            
            # Create mapping (label -> chunk_id)
            self.mapping = {str(i): chunk_id for i, chunk_id in enumerate(valid_chunk_ids)}
            
            # Save index
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            self.index.save_index(index_path)
            
            # Save mapping
            self._save_mapping(mapping_path, self.mapping)
            
            logger.info(
                f"HNSW index built successfully with {len(valid_embeddings)} vectors"
            )
            
            return {
                "success": True,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": len(valid_embeddings),
                "message": f"Index built successfully with {len(valid_embeddings)} vectors"
            }
            
        except Exception as e:
            logger.error(f"Error building HNSW index: {e}")
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": 0,
                "message": f"Index building failed: {str(e)}"
            }
    
    def load(self, index_path: str, mapping_path: str) -> bool:
        """
        Load existing HNSW index from disk.
        
        Args:
            index_path: Path to index file
            mapping_path: Path to mapping file
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load mapping
            self.mapping = self._load_mapping(mapping_path)
            
            # Load index
            self.index = hnswlib.Index(
                space=self.config.get("space", "cosine"),
                dim=self.embedding_dim
            )
            self.index.load_index(str(index_path))
            
            logger.info(f"HNSW index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HNSW index: {e}")
            return False
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query HNSW index for nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (labels, distances)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded or built")
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        available = int(getattr(self.index, "element_count", 0))
        if available <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        requested_k = max(1, min(int(top_k), available))

        # Root Cause vs Logic: hnswlib raises a contiguous-array error when k is
        # too large for the graph's effective search breadth. We clamp k to the
        # loaded element count and grow ef_search until the existing index can
        # satisfy the request, bounded to avoid infinite retry loops.
        ef_search = max(requested_k * 2, 50)
        max_ef = max(settings.HNSW_MAX_EF_SEARCH, ef_search, available)
        last_error = None
        while ef_search <= max_ef:
            try:
                self.index.set_ef(min(ef_search, max_ef))
                labels, distances = self.index.knn_query(query_embedding, k=requested_k)
                return labels[0], distances[0]
            except RuntimeError as exc:
                last_error = exc
                message = str(exc).lower()
                if "contiguous 2d array" not in message and "ef or m is too small" not in message:
                    raise
                ef_search *= 2
        raise last_error or RuntimeError("HNSW query failed")
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded index.
        
        Returns:
            Dict with index metadata
        """
        if self.index is None:
            return {
                "type": "hnsw",
                "loaded": False,
                "message": "Index not loaded"
            }
        
        return {
            "type": "hnsw",
            "loaded": True,
            "dimension": self.embedding_dim,
            "total_vectors": self.index.element_count,
            "space": self.config.get("space", "cosine"),
            "M": self.config.get("M", 16),
            "ef_construction": self.config.get("ef_construction", 200)
        }


def load_hnsw_index(embedding_dim: int, index_path: str, mapping_path: str) -> HNSWIndexBuilder:
    """
    Convenience function to load HNSW index.
    
    Args:
        embedding_dim: Dimension of embeddings
        index_path: Path to index file
        mapping_path: Path to mapping file
        
    Returns:
        Loaded HNSWIndexBuilder instance
    """
    builder = HNSWIndexBuilder(embedding_dim)
    
    if not builder.load(index_path, mapping_path):
        raise RuntimeError(f"Failed to load HNSW index from {index_path}")
    
    return builder
