"""Reranker client adapter with retries and Azure/Cohere provider support."""

import httpx
import logging
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit

logger = logging.getLogger(__name__)


def _foundry_cohere_rerank_url() -> str:
    """Resolve Cohere's v2 rerank route from the configured Foundry account."""
    explicit = (os.getenv("CLOUD_RERANKER_URI") or "").strip()
    if explicit:
        return explicit
    endpoint = (settings.AZURE_AI_FOUNDRY_ENDPOINT or "").strip().rstrip("/")
    if not endpoint:
        return ""
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return f"{endpoint}/providers/cohere/v2/rerank"


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
        if settings.CLOUD_MODE:
            base_url = _foundry_cohere_rerank_url() or base_url
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
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank passages for a query."""
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
                "return_logits": return_logits,
            }

        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            logger.debug("Calling reranker at %s for %s passages", self.base_url, len(passages))
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers,
                fail_open_after_s=120.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            if self.provider == "cohere" and "results" in data:
                for item in data["results"]:
                    score = float(item.get("relevance_score", item.get("score", 0.0)))
                    results.append(
                        {
                            "index": item.get("index", 0),
                            "score": score,
                            "p_hat": score,
                            "calibration_version": "identity:cohere-v2",
                        }
                    )
            elif "results" in data:
                for item in data["results"]:
                    results.append(
                        {
                            "index": item.get("index", 0),
                            "score": item.get("score", 0.0),
                            "logit": item.get("logit"),
                            "p_hat": item.get("p_hat", item.get("score", 0.0)),
                            "calibration_version": item.get("calibration_version"),
                        }
                    )
            elif "scores" in data:
                for idx, score in enumerate(data["scores"]):
                    results.append({"index": idx, "score": float(score), "p_hat": float(score)})
            else:
                for idx, score in enumerate(data):
                    results.append({"index": idx, "score": float(score), "p_hat": float(score)})

            results.sort(key=lambda x: x["score"], reverse=True)
            if len(results) != len(passages):
                raise RuntimeError(
                    f"Reranker returned {len(results)} results for {len(passages)} passages"
                )
            return results
        except httpx.HTTPError as exc:
            logger.error("Reranker request failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error in reranker call: %s", exc)
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
