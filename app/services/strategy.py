"""
Index Strategy Manager for Dynamic Index Selection

This module manages different indexing strategies (HNSW, FAISS, Tree-based)
based on data characteristics and query requirements.
"""
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Available index types."""
    HNSW = "hnsw"
    FAISS_IVF = "faiss_ivf"
    FAISS_TREE = "faiss_tree"


class IndexStrategy(Enum):
    """Index selection strategies based on data characteristics."""
    # Single strategies
    HNSW_ONLY = "hnsw_only"
    FAISS_ONLY = "faiss_only"
    TREE_ONLY = "tree_only"
    
    # Hybrid strategies
    HNSW_FAISS = "hnsw_faiss"  # Use HNSW for retrieval, FAISS for reranking
    FAISS_HNSW = "faiss_hnsw"  # Use FAISS for coarse, HNSW for refinement
    TREE_HNSW = "tree_hnsw"    # Use Tree for structured, HNSW for semantic


@dataclass
class ChunkCharacteristics:
    """Characteristics of a chunk used for strategy selection."""
    content_type: str
    token_count: int
    chunk_length: int
    source: str
    has_metadata: bool = True


@dataclass
class QueryCharacteristics:
    """Characteristics of a query used for strategy selection."""
    query_length: int
    top_k: int
    use_reranking: bool
    has_filters: bool = False
    priority: str = "accuracy"  # "accuracy", "speed", "balanced"


class IndexStrategyManager:
    """
    Manages dynamic index selection based on data and query characteristics.
    """
    
    def __init__(self):
        """Initialize the strategy manager."""
        self.strategy_cache = {}
    
    def select_ingestion_strategy(
        self, 
        chunk_characteristics: ChunkCharacteristics,
        dataset_size: int
    ) -> IndexStrategy:
        """
        Select indexing strategy during ingestion based on chunk characteristics.
        
        Strategy selection rules:
        1. Large datasets (>100k) or fast ingestion needed → FAISS_IVF
        2. Small structured queries (question_part) → FAISS_TREE
        3. Complete dialogues or complex content → HNSW
        4. Hybrid approaches for mixed content types
        
        Args:
            chunk_characteristics: Characteristics of the chunk
            dataset_size: Total size of the dataset
            
        Returns:
            Recommended index strategy
        """
        content_type = chunk_characteristics.content_type
        token_count = chunk_characteristics.token_count
        source = chunk_characteristics.source
        
        # Rule 1: Large datasets favor FAISS for faster ingestion
        if dataset_size > 100000:
            logger.info(f"Large dataset ({dataset_size}), using FAISS_IVF for ingestion speed")
            return IndexStrategy.FAISS_ONLY
        
        # Rule 2: Small token counts with structured content → Tree
        if token_count < 200 and content_type in ["question_part1", "answer_part1"]:
            logger.info(f"Small structured chunk ({token_count} tokens, {content_type}), using Tree")
            return IndexStrategy.TREE_ONLY
        
        # Rule 3: Complete dialogues or complex content → HNSW
        if content_type in ["complete_dialogue", "question_final_and_answer"]:
            logger.info(f"Complete dialogue chunk ({content_type}), using HNSW")
            return IndexStrategy.HNSW_ONLY
        
        # Rule 4: Question parts for reranking scenarios → Hybrid
        if content_type in ["question_part1", "question_and_answer_part1"]:
            logger.info(f"Question chunk ({content_type}), using HNSW for initial retrieval")
            return IndexStrategy.HNSW_FAISS
        
        # Default: Use HNSW for balanced performance
        logger.info(f"Default strategy: HNSW for chunk {content_type}")
        return IndexStrategy.HNSW_ONLY
    
    def select_retrieval_strategy(
        self,
        query_characteristics: QueryCharacteristics
    ) -> IndexStrategy:
        """
        Select indexing strategy during retrieval based on query characteristics.
        
        Strategy selection rules:
        1. Speed priority + high top_k → FAISS
        2. Accuracy priority + reranking → HNSW
        3. Small top_k + filters → Tree
        4. Balanced approach → Hybrid strategies
        
        Args:
            query_characteristics: Characteristics of the query
            
        Returns:
            Recommended index strategy
        """
        top_k = query_characteristics.top_k
        use_reranking = query_characteristics.use_reranking
        priority = query_characteristics.priority
        has_filters = query_characteristics.has_filters
        
        # Rule 1: Speed priority and high top_k → FAISS
        if priority == "speed" and top_k > 50:
            logger.info(f"Speed priority with high top_k ({top_k}), using FAISS")
            return IndexStrategy.FAISS_ONLY
        
        # Rule 2: Reranking enabled → Use HNSW for initial retrieval
        if use_reranking:
            logger.info(f"Reranking enabled, using HNSW for initial retrieval")
            return IndexStrategy.HNSW_FAISS
        
        # Rule 3: Small top_k with filters → Tree
        if top_k < 5 and has_filters:
            logger.info(f"Small top_k ({top_k}) with filters, using Tree")
            return IndexStrategy.TREE_ONLY
        
        # Rule 4: Large top_k without reranking → Need efficiency
        if top_k > 20 and not use_reranking:
            logger.info(f"Large top_k ({top_k}) without reranking, using FAISS_IVF")
            return IndexStrategy.FAISS_ONLY
        
        # Rule 5: Balanced approach → Default to HNSW
        if priority == "balanced":
            logger.info(f"Balanced priority, using HNSW")
            return IndexStrategy.HNSW_ONLY
        
        # Default: Accuracy priority → HNSW
        logger.info(f"Default strategy: HNSW for query retrieval")
        return IndexStrategy.HNSW_ONLY
    
    def get_index_config(
        self,
        strategy: IndexStrategy,
        embedding_dim: int,
        dataset_size: int
    ) -> Dict[str, Any]:
        """
        Get configuration for building a specific index type.
        
        Args:
            strategy: The index strategy to configure
            embedding_dim: Dimension of embeddings
            dataset_size: Total size of the dataset
            
        Returns:
            Configuration dict for building the index
        """
        config = {
            "embedding_dim": embedding_dim,
            "dataset_size": dataset_size
        }
        
        # HNSW configuration
        if strategy in [IndexStrategy.HNSW_ONLY, IndexStrategy.HNSW_FAISS, 
                       IndexStrategy.FAISS_HNSW, IndexStrategy.TREE_HNSW]:
            config["hnsw"] = {
                "M": 16 if dataset_size < 50000 else 32,
                "ef_construction": 200,
                "space": "cosine",
                "max_elements": dataset_size * 2  # Allow for growth
            }
        
        # FAISS IVF configuration
        if strategy in [IndexStrategy.FAISS_ONLY, IndexStrategy.HNSW_FAISS,
                       IndexStrategy.FAISS_HNSW]:
            # Calculate number of clusters based on dataset size
            n_clusters = max(100, min(dataset_size // 1000, 8192))
            config["faiss_ivf"] = {
                "n_clusters": n_clusters,
                "n_probe": min(100, n_clusters // 10),
                "train_fraction": 0.1
            }
        
        # FAISS Tree configuration
        if strategy in [IndexStrategy.TREE_ONLY, IndexStrategy.TREE_HNSW]:
            config["faiss_tree"] = {
                "n_trees": 8 if dataset_size < 100000 else 16,
                "depth": 10,
                "budget": 100
            }
        
        return config
    
    def get_index_file_path(
        self,
        index_type: IndexType,
        source: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get file paths for index files based on index type.
        
        Args:
            index_type: Type of index
            source: Optional source identifier for named indexes
            
        Returns:
            Dict with index_path and mapping_path
        """
        base_dir = Path(settings.DATA_DIR)
        
        if source:
            index_filename = f"{index_type.value}_index_{source}.bin"
            mapping_filename = f"{index_type.value}_mapping_{source}.json"
        else:
            index_filename = f"{index_type.value}_index.bin"
            mapping_filename = f"{index_type.value}_mapping.json"
        
        return {
            "index_path": str(base_dir / index_filename),
            "mapping_path": str(base_dir / mapping_filename)
        }
    
    def should_rebuild_index(
        self,
        existing_index_path: str,
        new_chunk_count: int,
        current_index_size: int,
        rebuild_threshold: float = 0.2  # 20% growth threshold
    ) -> bool:
        """
        Determine if index should be rebuilt based on growth.
        
        Args:
            existing_index_path: Path to existing index
            new_chunk_count: Number of new chunks to add
            current_index_size: Current size of existing index
            rebuild_threshold: Fraction growth that triggers rebuild
            
        Returns:
            True if index should be rebuilt
        """
        if not Path(existing_index_path).exists():
            return True
        
        growth_ratio = new_chunk_count / current_index_size if current_index_size > 0 else 1.0
        
        return growth_ratio >= rebuild_threshold


def analyze_chunk_characteristics(chunk: Dict[str, Any]) -> ChunkCharacteristics:
    """
    Extract characteristics from a chunk for strategy selection.
    
    Args:
        chunk: Chunk dictionary with metadata
        
    Returns:
        ChunkCharacteristics object
    """
    metadata = chunk.get("metadata", {})
    
    return ChunkCharacteristics(
        content_type=metadata.get("content_type", "complete_dialogue"),
        token_count=metadata.get("token_count", 0),
        chunk_length=metadata.get("chunk_length", 0),
        source=metadata.get("source", "unknown"),
        has_metadata=bool(metadata)
    )


def analyze_query_characteristics(
    query: str,
    top_k: int,
    use_reranking: bool,
    filters: Optional[Dict[str, Any]] = None
) -> QueryCharacteristics:
    """
    Extract characteristics from a query for strategy selection.
    
    Args:
        query: Query string
        top_k: Number of results requested
        use_reranking: Whether reranking will be used
        filters: Optional metadata filters
        
    Returns:
        QueryCharacteristics object
    """
    return QueryCharacteristics(
        query_length=len(query.split()),
        top_k=top_k,
        use_reranking=use_reranking,
        has_filters=bool(filters),
        priority="balanced"  # Could be extracted from request if needed
    )

