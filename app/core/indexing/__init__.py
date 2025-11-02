"""Index builders and loaders for different index types."""

from app.core.indexing.base import BaseIndexBuilder
from app.core.indexing.hnsw import HNSWIndexBuilder, load_hnsw_index
from app.core.indexing.tree import TreeIndexBuilder, load_tree_index

# Import FAISS modules with error handling
try:
    from app.core.indexing.faiss import FAISSIndexBuilder, load_faiss_ivf_index
    FAISS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    FAISS_AVAILABLE = False
    # Create dummy classes if FAISS is not available
    class FAISSIndexBuilder:
        pass
    def load_faiss_ivf_index(*args, **kwargs):
        raise ImportError("FAISS is not available")

__all__ = [
    "BaseIndexBuilder",
    "HNSWIndexBuilder",
    "load_hnsw_index",
    "FAISSIndexBuilder",
    "load_faiss_ivf_index",
    "TreeIndexBuilder",
    "load_tree_index",
]

