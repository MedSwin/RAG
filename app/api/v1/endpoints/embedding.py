from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import logging

from app.core.state import get_model_manager
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

class EmbeddingRequest(BaseModel):
    """Request model for text embedding."""
    text: str
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    """Response model for text embedding."""
    embedding: List[float]
    dimension: int
    model_name: str

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding."""
    texts: List[str]
    normalize: bool = True
    batch_size: int = 64

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding."""
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    count: int

def get_embedding_model():
    """Dependency to get embedding model."""
    try:
        return get_model_manager().get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to get embedding model: {e}")
        raise HTTPException(status_code=503, detail="Embedding model not available")

@router.post("/embed", response_model=EmbeddingResponse)
async def embed_text(
    request: EmbeddingRequest,
    model_components = Depends(get_embedding_model)
):
    """Embed a single text."""
    try:
        tokenizer, embed_model, device, embedding_dim = model_components
        
        # Tokenize input
        inputs = tokenizer(
            [request.text],
            truncation=True,
            padding=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = embed_model(**inputs)
            attention_mask = inputs['attention_mask']
            embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
            
            if request.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
            
            embedding = embedding.cpu().numpy().astype(np.float64)
        
        return EmbeddingResponse(
            embedding=embedding[0].tolist(),
            dimension=embedding_dim,
            model_name=settings.EMBEDDING_MODEL_PATH
        )
        
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@router.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_texts_batch(
    request: BatchEmbeddingRequest,
    model_components = Depends(get_embedding_model)
):
    """Embed multiple texts in batch."""
    try:
        tokenizer, embed_model, device, embedding_dim = model_components
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(request.texts), request.batch_size):
            batch_texts = request.texts[i:i + request.batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = embed_model(**inputs)
                attention_mask = inputs['attention_mask']
                batch_embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
                
                if request.normalize:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float64)
                all_embeddings.extend(batch_embeddings.tolist())
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return BatchEmbeddingResponse(
            embeddings=all_embeddings,
            dimension=embedding_dim,
            model_name=settings.EMBEDDING_MODEL_PATH,
            count=len(all_embeddings)
        )
        
    except Exception as e:
        logger.error(f"Error embedding texts batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

@router.get("/info")
async def get_embedding_info(model_components = Depends(get_embedding_model)):
    """Get embedding model information."""
    try:
        tokenizer, embed_model, device, embedding_dim = model_components
        
        return {
            "model_path": settings.EMBEDDING_MODEL_PATH,
            "dimension": embedding_dim,
            "device": str(device),
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
            "model_type": type(embed_model).__name__
        }
        
    except Exception as e:
        logger.error(f"Error getting embedding info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling function."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = summed_embeddings / summed_mask
    return mean_embeddings
