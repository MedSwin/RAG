"""Reranker client adapter with retries and timeouts."""

import httpx
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.adapters.rate_limit import request_with_model_rate_limit

logger = logging.getLogger(__name__)


class RerankerClient:
    """Client for reranker endpoints."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """Initialize reranker client.
        
        Args:
            base_url: Base URL for the reranker endpoint
            timeout: Timeout in seconds (defaults to RERANK_TIMEOUT_S)
        """
        self.base_url = base_url
        self.timeout = timeout or settings.RERANK_TIMEOUT_S
        self.model = model or (settings.CLOUD_RERANKER if settings.CLOUD_MODE else "default")
        self.api_key = api_key or (settings.AZURE_AI_FOUNDRY_API_KEY if settings.CLOUD_MODE else None)
        self.provider = provider or ("cohere" if settings.CLOUD_MODE else "default")
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"reranker:{self.base_url}:{self.model}"
    
    async def rerank(
        self,
        query: str,
        passages: List[str],
        return_logits: bool = False,
        request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rerank passages for a query.
        
        Args:
            query: Query text
            passages: List of passage texts to rerank
            return_logits: Whether to return raw logits
            request_id: Optional request ID for tracing
            
        Returns:
            List of dicts with 'index', 'score', 'logit' (optional), 'p_hat' (calibrated probability)
            
        Raises:
            httpx.HTTPError: If request fails after retries
        """
        if not passages:
            return []
        
        if self.provider == "cohere":
            payload = {
                "model": self.model,
                "query": query,
                "documents": passages,
                "top_n": len(passages),
            }
        else:
            payload = {
                "query": query,
                "passages": passages,
                "return_logits": return_logits
            }
        
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id
        
        try:
            logger.debug(f"Calling reranker at {self.base_url} for {len(passages)} passages")
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract scores from response
            results = []
            if self.provider == "cohere" and "results" in data:
                for item in data["results"]:
                    score = float(item.get("relevance_score", item.get("score", 0.0)))
                    results.append({
                        "index": item.get("index", 0),
                        "score": score,
                        "p_hat": score,
                        "calibration_version": "identity:cohere-v2",
                    })
            elif "results" in data:
                # Structured format
                for item in data["results"]:
                    results.append({
                        "index": item.get("index", 0),
                        "score": item.get("score", 0.0),
                        "logit": item.get("logit"),
                        "p_hat": item.get("p_hat", item.get("score", 0.0)),
                        "calibration_version": item.get("calibration_version"),
                    })
            elif "scores" in data:
                # Simple scores array
                for idx, score in enumerate(data["scores"]):
                    results.append({
                        "index": idx,
                        "score": float(score),
                        "p_hat": float(score)  # Assume score is already calibrated
                    })
            else:
                # Assume direct list of scores
                for idx, score in enumerate(data):
                    results.append({
                        "index": idx,
                        "score": float(score),
                        "p_hat": float(score)
                    })
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
            
        except httpx.HTTPError as e:
            logger.error(f"Reranker request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in reranker call: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
