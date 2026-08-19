"""Embedding API aligned with the active local/cloud runtime."""

from __future__ import annotations

import logging
from typing import List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.state import get_model_manager
from app.services.adapters.embedding import EmbeddingClient

logger = logging.getLogger(__name__)
router = APIRouter()

EmbeddingIntent = Literal["query", "document"]


class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)
    normalize: bool = True
    input_type: EmbeddingIntent = "query"


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model_name: str
    input_type: EmbeddingIntent


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(min_length=1)
    normalize: bool = True
    batch_size: int = Field(default=64, ge=1, le=512)
    input_type: EmbeddingIntent = "query"


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    count: int
    input_type: EmbeddingIntent


def _get_local_embedding_model():
    try:
        return get_model_manager().get_embedding_model()
    except Exception as exc:
        logger.error("Failed to get local embedding model: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding model not available")


async def _cloud_embed(texts: List[str], input_type: EmbeddingIntent) -> List[List[float]]:
    client = EmbeddingClient(settings.active_embedding_url())
    try:
        vectors = await client.embed(texts, input_type=input_type)
    finally:
        await client.close()
    return [vector.astype(np.float32).tolist() for vector in vectors]


def _local_embed(texts: List[str], normalize: bool) -> tuple[List[List[float]], int]:
    tokenizer, embed_model, device, embedding_dim = _get_local_embedding_model()
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=settings.MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embedding = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        array = embedding.cpu().numpy().astype(np.float32)
    return array.tolist(), int(embedding_dim)


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    try:
        if settings.CLOUD_MODE:
            vectors = await _cloud_embed([request.text], request.input_type)
            if len(vectors) != 1:
                raise RuntimeError("Embedding service did not return exactly one vector")
            return EmbeddingResponse(
                embedding=vectors[0],
                dimension=len(vectors[0]),
                model_name=settings.active_embedding_model(),
                input_type=request.input_type,
            )

        vectors, dimension = _local_embed([request.text], request.normalize)
        return EmbeddingResponse(
            embedding=vectors[0],
            dimension=dimension,
            model_name=settings.EMBEDDING_MODEL_PATH,
            input_type=request.input_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error embedding text: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")


@router.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_texts_batch(request: BatchEmbeddingRequest):
    try:
        all_embeddings: List[List[float]] = []
        dimension = settings.active_embedding_dimension() if settings.CLOUD_MODE else 0
        for start in range(0, len(request.texts), request.batch_size):
            batch = request.texts[start : start + request.batch_size]
            if settings.CLOUD_MODE:
                batch_embeddings = await _cloud_embed(batch, request.input_type)
                if batch_embeddings:
                    dimension = len(batch_embeddings[0])
            else:
                batch_embeddings, dimension = _local_embed(batch, request.normalize)
            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) != len(request.texts):
            raise RuntimeError(
                f"Embedding runtime returned {len(all_embeddings)} vectors for {len(request.texts)} inputs"
            )
        return BatchEmbeddingResponse(
            embeddings=all_embeddings,
            dimension=dimension,
            model_name=settings.active_embedding_model() if settings.CLOUD_MODE else settings.EMBEDDING_MODEL_PATH,
            count=len(all_embeddings),
            input_type=request.input_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error embedding text batch: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {exc}")


@router.get("/info")
async def get_embedding_info():
    """Return active embedding identity without forcing unavailable local models."""
    if settings.CLOUD_MODE:
        return {
            "model_path": settings.active_embedding_url(),
            "model_name": settings.active_embedding_model(),
            "dimension": settings.active_embedding_dimension(),
            "device": "azure-foundry",
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
            "model_type": "cloud",
        }
    tokenizer, embed_model, device, embedding_dim = _get_local_embedding_model()
    return {
        "model_path": settings.EMBEDDING_MODEL_PATH,
        "model_name": settings.active_embedding_model(),
        "dimension": embedding_dim,
        "device": str(device),
        "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
        "model_type": type(embed_model).__name__,
        "tokenizer_type": type(tokenizer).__name__,
    }


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return summed_embeddings / summed_mask
