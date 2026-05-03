"""Embedding client adapter with retries and timeouts."""

import httpx
import logging
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
from app.core.config import settings
import numpy as np

logger = logging.getLogger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code in {429, 500, 502, 503, 504}
    )


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
    
    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=(
            retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
            | retry_if_exception(_is_retryable_http_error)
        ),
        reraise=True
    )
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
            response = await self.client.post(
                self.base_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            if "data" in data:
                # OpenAI-compatible format
                for item in data["data"]:
                    if "embedding" in item:
                        embeddings.append(np.array(item["embedding"], dtype=np.float32))
            elif "embeddings" in data:
                # Alternative format
                embeddings = [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]
            else:
                # Assume direct list of embeddings
                embeddings = [np.array(emb, dtype=np.float32) for emb in data]
            
            return embeddings
            
        except httpx.HTTPError as e:
            logger.error(f"Embedding request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding call: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
