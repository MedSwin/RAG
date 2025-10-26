import torch
import logging
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.services.reranker_service import DocumentReranker

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of ML models."""
    
    def __init__(self):
        self.embedding_tokenizer: Optional[AutoTokenizer] = None
        self.embedding_model: Optional[AutoModel] = None
        self.device: Optional[torch.device] = None
        self.embedding_dimension: Optional[int] = None
        self.reranker_model: Optional[DocumentReranker] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def load_embedding_model(self):
        """Load embedding model asynchronously."""
        try:
            model_path = Path(settings.EMBEDDING_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Embedding model not found at {model_path}")
            
            logger.info(f"Loading embedding model from {model_path}")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._load_embedding_model_sync, 
                model_path
            )
            
            self.embedding_tokenizer, self.embedding_model, self.device, self.embedding_dimension = result
            logger.info(f"Embedding model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_embedding_model_sync(self, model_path: Path) -> Tuple[AutoTokenizer, AutoModel, torch.device, int]:
        """Synchronous model loading function."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        # Detect embedding dimension
        test_input = tokenizer("test", return_tensors="pt", truncation=True, padding=True, max_length=512)
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            test_output = model(**test_input)
            test_embedding = self._mean_pooling(test_output.last_hidden_state, test_input['attention_mask'])
            embedding_dim = test_embedding.shape[1]
        
        return tokenizer, model, device, embedding_dim
    
    async def load_reranker_model(self):
        """Load reranker model asynchronously."""
        try:
            model_path = Path(settings.RERANKER_MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"Reranker model not found at {model_path}, skipping...")
                return
            
            logger.info(f"Loading reranker model from {model_path}")
            
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            self.reranker_model = await loop.run_in_executor(
                self.executor,
                self._load_reranker_model_sync,
                model_path
            )
            
            logger.info("Reranker model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            # Don't raise exception as reranker is optional
    
    def _load_reranker_model_sync(self, model_path: Path) -> DocumentReranker:
        """Synchronous reranker loading function."""
        return DocumentReranker(str(model_path))
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling function."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * input_mask_expanded
        summed_embeddings = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = summed_embeddings / summed_mask
        return mean_embeddings
    
    def get_embedding_model(self) -> Tuple[AutoTokenizer, AutoModel, torch.device, int]:
        """Get embedding model components."""
        if not all([self.embedding_tokenizer, self.embedding_model, self.device, self.embedding_dimension]):
            raise Exception("Embedding model not loaded")
        return self.embedding_tokenizer, self.embedding_model, self.device, self.embedding_dimension
    
    def get_reranker_model(self) -> Optional[DocumentReranker]:
        """Get reranker model."""
        return self.reranker_model
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model manager cleaned up")
