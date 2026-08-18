"""Embedding client adapter with retries and timeouts."""

import asyncio
import httpx
import logging
from typing import List, Optional
from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for embedding endpoints."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize embedding client.
        
        Args:
            base_url: Base URL for the embedding endpoint
            timeout: Timeout in seconds (defaults to EMBED_TIMEOUT_S)
        """
        self.base_url = base_url
        self.timeout = timeout or settings.EMBED_TIMEOUT_S
        self.model = model or (settings.CLOUD_EMBEDDING if settings.CLOUD_MODE else "default")
        self.api_key = api_key or (settings.AZURE_AI_FOUNDRY_API_KEY if settings.CLOUD_MODE else None)
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"embedding:{self.base_url}:{self.model}"
    
    async def embed(self, texts: List[str], request_id: Optional[str] = None) -> List[np.ndarray]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            request_id: Optional request ID for tracing
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            httpx.HTTPError: If request fails after retries
        """
        if not texts:
            return []
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id
        
        try:
            logger.debug(f"Calling embedding service at {self.base_url} for {len(texts)} texts")
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return _parse_embedding_response(response.json())
        except Exception as exc:
            if settings.CLOUD_MODE:
                logger.error(f"Embedding request failed: {exc}")
                raise
            local = await self._local_embed(texts)
            if local is not None:
                logger.warning(
                    "HTTP embedding at %s failed (%s); using locally loaded ModelManager",
                    self.base_url,
                    exc,
                )
                return local
            logger.error(f"Embedding request failed: {exc}")
            raise

    async def _local_embed(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Use the process-local embedding model when the HTTP endpoint is down."""
        try:
            from app.core.state import get_model_manager
            import torch
            import torch.nn.functional as F

            tokenizer, embed_model, device, _dim = get_model_manager().get_embedding_model()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Local embedding model is not available: %s", exc)
            return None

        loop = asyncio.get_running_loop()

        def _encode() -> List[np.ndarray]:
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
                mask = inputs["attention_mask"]
                pooled = _mean_pool(outputs.last_hidden_state, mask)
                pooled = F.normalize(pooled, p=2, dim=1)
            return [row.astype(np.float32) for row in pooled.cpu().numpy()]

        return await loop.run_in_executor(None, _encode)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def _parse_embedding_response(data) -> List[np.ndarray]:
    embeddings = []
    if isinstance(data, dict) and "data" in data:
        for item in data["data"]:
            if "embedding" in item:
                embeddings.append(np.array(item["embedding"], dtype=np.float32))
    elif isinstance(data, dict) and "embeddings" in data:
        embeddings = [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]
    else:
        embeddings = [np.array(emb, dtype=np.float32) for emb in data]
    return embeddings


def _mean_pool(token_embeddings, attention_mask):
    import torch

    expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
