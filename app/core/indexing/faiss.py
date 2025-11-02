"""FAISS index builder and loader."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    import faiss
    # Test if faiss works by checking for required attributes
    _ = faiss.IndexFlatL2
    FAISS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    FAISS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"FAISS not available: {e}. Install with: pip install faiss-cpu or faiss-gpu")

from app.core.indexing.base import BaseIndexBuilder
from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSIndexBuilder(BaseIndexBuilder):
    """Builder for FAISS IVF (Inverted File Index) indexes."""
    
    def __init__(
        self,
        embedding_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FAISS IVF index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
            config: Configuration dict with n_clusters, n_probe, train_fraction
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        if config is None:
            config = {
                "n_clusters": 100,
                "n_probe": 10,
                "train_fraction": 0.1
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
        Build FAISS IVF index from embeddings.
        
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
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create quantizer
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            
            # Create IVF index
            n_clusters = self.config.get("n_clusters", 100)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            
            # Train the index
            train_fraction = self.config.get("train_fraction", 0.1)
            n_train = max(int(len(valid_embeddings) * train_fraction), n_clusters)
            n_train = min(n_train, len(valid_embeddings))
            
            logger.info(f"Training FAISS index with {n_train} vectors")
            index.train(embeddings_array[:n_train])
            
            # Add all vectors
            index.add(embeddings_array)
            
            # Set nprobe
            index.nprobe = self.config.get("n_probe", 10)
            
            self.index = index
            
            # Create mapping (label -> chunk_id)
            self.mapping = {str(i): chunk_id for i, chunk_id in enumerate(valid_chunk_ids)}
            
            # Save index
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, index_path)
            
            # Save mapping
            self._save_mapping(mapping_path, self.mapping)
            
            logger.info(
                f"FAISS index built successfully with {len(valid_embeddings)} vectors"
            )
            
            return {
                "success": True,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": len(valid_embeddings),
                "message": f"Index built successfully with {len(valid_embeddings)} vectors"
            }
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "total_vectors": 0,
                "message": f"Index building failed: {str(e)}"
            }
    
    def load(self, index_path: str, mapping_path: str) -> bool:
        """
        Load existing FAISS index from disk.
        
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
            self.index = faiss.read_index(str(index_path))
            
            # Set nprobe
            self.index.nprobe = self.config.get("n_probe", 10)
            
            logger.info(f"FAISS index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query FAISS index for nearest neighbors.
        
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
        
        # Normalize for cosine similarity
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Query
        distances, labels = self.index.search(query_embedding, top_k)
        
        return labels[0], distances[0]
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded index.
        
        Returns:
            Dict with index metadata
        """
        if self.index is None:
            return {
                "type": "faiss_ivf",
                "loaded": False,
                "message": "Index not loaded"
            }
        
        return {
            "type": "faiss_ivf",
            "loaded": True,
            "dimension": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "n_clusters": self.config.get("n_clusters", 100),
            "n_probe": self.config.get("n_probe", 10)
        }


def load_faiss_ivf_index(
    embedding_dim: int,
    index_path: str,
    mapping_path: str
) -> FAISSIndexBuilder:
    """
    Convenience function to load FAISS IVF index.
    
    Args:
        embedding_dim: Dimension of embeddings
        index_path: Path to index file
        mapping_path: Path to mapping file
        
    Returns:
        Loaded FAISSIndexBuilder instance
    """
    builder = FAISSIndexBuilder(embedding_dim)
    
    if not builder.load(index_path, mapping_path):
        raise RuntimeError(f"Failed to load FAISS index from {index_path}")
    
    return builder

