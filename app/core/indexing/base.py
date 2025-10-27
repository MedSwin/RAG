"""Base class for index builders."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseIndexBuilder(ABC):
    """
    Base class for all index builders.
    
    All index builders must implement these methods:
    - build: Build index from embeddings
    - load: Load existing index
    - query: Query index with embeddings
    """
    
    def __init__(self, embedding_dim: int, config: Dict[str, Any]):
        """
        Initialize index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
            config: Configuration dictionary for the index
        """
        self.embedding_dim = embedding_dim
        self.config = config
        self.index = None
    
    @abstractmethod
    def build(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        index_path: str,
        mapping_path: str
    ) -> Dict[str, Any]:
        """
        Build index from embeddings.
        
        Args:
            embeddings: List of embeddings
            chunk_ids: List of chunk IDs corresponding to embeddings
            index_path: Path to save index file
            mapping_path: Path to save mapping file
            
        Returns:
            Dict with success status and metadata
        """
        pass
    
    @abstractmethod
    def load(self, index_path: str, mapping_path: str) -> bool:
        """
        Load existing index from disk.
        
        Args:
            index_path: Path to index file
            mapping_path: Path to mapping file
            
        Returns:
            True if loaded successfully
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query index for nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (labels, distances)
        """
        pass
    
    @abstractmethod
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded index.
        
        Returns:
            Dict with index metadata
        """
        pass
    
    def _save_mapping(self, mapping_path: str, mapping: Dict[str, str]):
        """Save index mapping to JSON file."""
        Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f)
        logger.info(f"Saved mapping to {mapping_path}")
    
    def _load_mapping(self, mapping_path: str) -> Dict[str, str]:
        """Load index mapping from JSON file."""
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded mapping from {mapping_path}")
        return mapping
    
    def _validate_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str]
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Validate embeddings and filter out invalid ones.
        
        Args:
            embeddings: List of embeddings
            chunk_ids: List of chunk IDs
            
        Returns:
            Tuple of (valid_embeddings, valid_chunk_ids)
        """
        valid_embeddings = []
        valid_chunk_ids = []
        
        for i, (emb, chunk_id) in enumerate(zip(embeddings, chunk_ids)):
            if len(emb) == self.embedding_dim:
                valid_embeddings.append(emb)
                valid_chunk_ids.append(chunk_id)
            else:
                logger.warning(
                    f"Chunk {chunk_id} has invalid embedding dimension: "
                    f"{len(emb)} (expected {self.embedding_dim})"
                )
        
        return valid_embeddings, valid_chunk_ids

