"""Tree-based index builder and loader."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from sklearn.neighbors import BallTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

from app.core.indexing.base import BaseIndexBuilder
from app.core.config import settings

logger = logging.getLogger(__name__)


class TreeIndexBuilder(BaseIndexBuilder):
    """Builder for Tree-based (Ball Tree) indexes for structured queries."""
    
    def __init__(
        self,
        embedding_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Tree index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
            config: Configuration dict with leaf_size
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        
        if config is None:
            config = {
                "leaf_size": 40,
                "metric": "cosine"
            }
        
        super().__init__(embedding_dim, config)
        self.index = None
        self.mapping = {}
        self.embeddings_array = None
    
    def build(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        index_path: str,
        mapping_path: str
    ) -> Dict[str, Any]:
        """
        Build Tree index from embeddings.
        
        Args:
            embeddings: List of embeddings
            chunk_ids: List of chunk IDs corresponding to embeddings
            index_path: Path to save index file (not used for BallTree, but kept for consistency)
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
            self.embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            
            # Build Ball Tree
            leaf_size = self.config.get("leaf_size", 40)
            metric = self.config.get("metric", "cosine")
            
            logger.info(f"Building Ball Tree with leaf_size={leaf_size}, metric={metric}")
            self.index = BallTree(
                self.embeddings_array,
                leaf_size=leaf_size,
                metric=metric
            )
            
            # Create mapping (label -> chunk_id)
            self.mapping = {str(i): chunk_id for i, chunk_id in enumerate(valid_chunk_ids)}
            
            # Save mapping (BallTree doesn't support persistence, so we only save mapping)
            self._save_mapping(mapping_path, self.mapping)
            
            # Save embeddings for reloading (since BallTree doesn't support persistence)
            if index_path:
                Path(index_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(index_path, self.embeddings_array)
            
            logger.info(
                f"Tree index built successfully with {len(valid_embeddings)} vectors"
            )
            
            return {
                "success": True,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": len(valid_embeddings),
                "message": f"Index built successfully with {len(valid_embeddings)} vectors"
            }
            
        except Exception as e:
            logger.error(f"Error building Tree index: {e}")
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": 0,
                "message": f"Index building failed: {str(e)}"
            }
    
    def load(self, index_path: str, mapping_path: str) -> bool:
        """
        Load existing Tree index from disk.
        
        Args:
            index_path: Path to embeddings array file
            mapping_path: Path to mapping file
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load mapping
            self.mapping = self._load_mapping(mapping_path)
            
            # Load embeddings array
            if index_path and Path(index_path).exists():
                self.embeddings_array = np.load(index_path)
                
                # Rebuild tree
                leaf_size = self.config.get("leaf_size", 40)
                metric = self.config.get("metric", "cosine")
                
                self.index = BallTree(
                    self.embeddings_array,
                    leaf_size=leaf_size,
                    metric=metric
                )
            else:
                logger.warning("Embeddings file not found, cannot rebuild tree index")
                return False
            
            logger.info(f"Tree index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Tree index: {e}")
            return False
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query Tree index for nearest neighbors.
        
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
        
        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Query
        distances, labels = self.index.query(query_embedding, k=top_k)
        
        return labels[0], distances[0]
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded index.
        
        Returns:
            Dict with index metadata
        """
        if self.index is None:
            return {
                "type": "tree",
                "loaded": False,
                "message": "Index not loaded"
            }
        
        return {
            "type": "tree",
            "loaded": True,
            "dimension": self.embedding_dim,
            "total_vectors": len(self.embeddings_array) if self.embeddings_array is not None else 0,
            "leaf_size": self.config.get("leaf_size", 40),
            "metric": self.config.get("metric", "cosine")
        }


def load_tree_index(embedding_dim: int, index_path: str, mapping_path: str) -> TreeIndexBuilder:
    """
    Convenience function to load Tree index.
    
    Args:
        embedding_dim: Dimension of embeddings
        index_path: Path to embeddings array file
        mapping_path: Path to mapping file
        
    Returns:
        Loaded TreeIndexBuilder instance
    """
    builder = TreeIndexBuilder(embedding_dim)
    
    if not builder.load(index_path, mapping_path):
        raise RuntimeError(f"Failed to load Tree index from {index_path}")
    
    return builder

