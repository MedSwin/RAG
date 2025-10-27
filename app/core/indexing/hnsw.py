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
            
            self.index.init_index(
                max_elements=self.config.get("max_elements", len(valid_embeddings)),
                ef_construction=self.config.get("ef_construction", 200),
                M=self.config.get("M", 16)
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
        
        # Set ef_search based on top_k
        ef_search = max(top_k * 2, 50)
        self.index.set_ef(ef_search)
        
        # Query
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        
        return labels[0], distances[0]
    
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

