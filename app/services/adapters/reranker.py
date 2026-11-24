"""Reranker client adapter with retries and Azure/Cohere provider support."""

import asyncio
import httpx
import logging
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit

logger = logging.getLogger(__name__)


def _walk_exceptions(exc: BaseException):
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.append(getattr(current, "__cause__", None))
        stack.append(getattr(current, "__context__", None))


def _connection_failure(exc: BaseException) -> bool:
    """Return True for a dead socket, a connect timeout, or a refused connection."""
    connect_types = (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)
    for current in _walk_exceptions(exc):
        if isinstance(current, connect_types):
            return True
    text = str(exc).lower()
    markers = (
        "connection refused",
        "connecterror",
        "connection error",
        "all connection attempts failed",
        "name or service not known",
        "nodename nor servname",
    )
    return any(marker in text for marker in markers)


def _stick_to_local(exc: BaseException) -> bool:
    """Remember a dead reranker port. A connect timeout may still recover."""
    for current in _walk_exceptions(exc):
        if isinstance(current, httpx.ConnectTimeout):
            return False
        if isinstance(current, (httpx.ConnectError, ConnectionError)):
            return True
    text = str(exc).lower()
    return "connection refused" in text or "all connection attempts failed" in text


def _relevance_scores(reranker: Any, query: str, passages: List[str]) -> List[float]:
    """Call the batch scorer, or score each passage when the model is older."""
    batch = getattr(reranker, "compute_relevance_scores", None)
    if callable(batch):
        return [float(score) for score in batch(query, passages)]
    single = getattr(reranker, "compute_relevance_score", None)
    if not callable(single):
        raise RuntimeError("Local reranker does not implement compute_relevance_scores")
    return [float(single(query, passage)) for passage in passages]


def _cohere_auth_headers(api_key: Optional[str]) -> dict[str, str]:
    """Build an explicit authentication contract for Cohere-compatible routes.

    Azure/Foundry deployments can be exposed through provider-native routes or
    deployment front doors with different accepted header schemes. Publication
    evaluation must not guess silently, so the default matches Cohere ClientV2
    (Bearer) while an operator can explicitly select ``api-key`` or raw
    ``Authorization`` when their Foundry deployment requires it.
    """
    if not api_key:
        return {}
    scheme = (os.getenv("CLOUD_RERANKER_AUTH_SCHEME") or "bearer").strip().lower()
    if scheme in {"bearer", "token"}:
        return {"Authorization": f"Bearer {api_key}"}
    if scheme in {"api-key", "api_key", "apikey"}:
        return {"api-key": api_key}
    if scheme in {"raw", "authorization"}:
        return {"Authorization": api_key}
    raise ValueError(f"Unsupported CLOUD_RERANKER_AUTH_SCHEME={scheme}")


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
        if settings.CLOUD_MODE and settings.cloud_provider() == "bedrock" and provider in {None, "bedrock"}:
            self.base_url = f"bedrock://{settings.AWS_BEDROCK_RERANK_REGION}"
            self.model = model or settings.BEDROCK_RERANK_MODEL
            self.api_key = None
            self.provider = "bedrock"
        elif settings.CLOUD_MODE and provider in {None, "cohere"}:
            base_url = settings.cloud_reranker_url() or base_url
            self.base_url = base_url
            self.model = model or settings.CLOUD_RERANKER
            self.api_key = api_key or settings.AZURE_AI_FOUNDRY_API_KEY
            self.provider = provider or "cohere"
        else:
            self.base_url = base_url
            self.model = model or "default"
            self.api_key = api_key
            self.provider = provider or "default"
        self.timeout = timeout or settings.RERANK_TIMEOUT_S
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"reranker:{self.base_url}:{self.model}"
        self._prefer_inprocess_local = False

    def _wants_inprocess_local(self) -> bool:
        """Return True when local mode should score with the loaded bge reranker."""
        if settings.CLOUD_MODE:
            return False
        if str(self.base_url).startswith("local://"):
            return True
        flag = os.getenv("LOCAL_INPROCESS_MODELS", "").strip().lower()
        if flag in {"1", "true", "yes"}:
            return True
        return bool(getattr(self, "_prefer_inprocess_local", False))

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
        from app.services.medswin.governance import lead_with_caution, redact_phi_text

        query = redact_phi_text(query or "")
        passages = [lead_with_caution(redact_phi_text(text or "")) for text in passages]
        if self.provider == "bedrock":
            return await self._rerank_bedrock(query, passages)
        if self._wants_inprocess_local():
            local = await self._local_rerank(query, passages)
            if local is None:
                raise RuntimeError("In-process local reranker is unavailable")
            return local

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

        headers = _cohere_auth_headers(self.api_key) if self.provider == "cohere" else {}
        if self.api_key and self.provider != "cohere":
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
            if {int(item.get("index", -1)) for item in results} != set(range(len(passages))):
                raise RuntimeError("Reranker result indexes do not cover every supplied passage exactly once")
            return results
        except Exception as exc:
            if not settings.CLOUD_MODE and _connection_failure(exc):
                local = await self._local_rerank(query, passages)
                if local is not None:
                    if _stick_to_local(exc):
                        self._prefer_inprocess_local = True
                    logger.warning(
                        "HTTP reranker at %s failed (%s); using in-process bge-reranker-v2-m3",
                        self.base_url,
                        exc,
                    )
                    return local
            logger.error("Reranker request failed: %s", exc)
            raise

    async def _rerank_bedrock(self, query: str, passages: List[str]) -> List[Dict[str, Any]]:
        import asyncio

        from app.services.adapters.bedrock import BedrockRuntime, parse_rerank_response, rerank_request

        body = rerank_request(query, passages)
        runtime = getattr(self, "bedrock_runtime", None) or BedrockRuntime(settings.AWS_BEDROCK_RERANK_REGION)
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, lambda: runtime.invoke_model(self.model, body))
        return parse_rerank_response(payload, len(passages))

    async def _local_rerank(self, query: str, passages: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Score passages with the process-local ``bge-reranker-v2-m3`` model."""
        try:
            from app.core.state import get_model_manager

            reranker = get_model_manager().get_reranker_model()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Local reranker model is not available: %s", exc)
            return None
        if reranker is None:
            return None

        loop = asyncio.get_running_loop()

        def _score() -> List[float]:
            return _relevance_scores(reranker, query, passages)

        try:
            scores = await loop.run_in_executor(None, _score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("In-process reranker scoring failed: %s", exc)
            return None
        if scores is None or len(scores) != len(passages):
            logger.warning(
                "In-process reranker returned %s scores for %s passages",
                None if scores is None else len(scores),
                len(passages),
            )
            return None
        results = [
            {
                "index": idx,
                "score": float(score),
                "p_hat": float(score),
                "calibration_version": "identity:bge-reranker-v2-m3-local",
            }
            for idx, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
