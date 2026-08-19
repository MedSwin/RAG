"""Embedding client adapter with retries, provider-aware payloads, and local fallback.

Azure Foundry can expose Cohere Embed v4 through more than one wire contract:

* Foundry model-inference/OpenAI-style embeddings: ``input`` + optional
  ``query``/``document`` intent.
* Cohere-native deployment endpoints such as ``.../v1/embed`` or ``.../v2/embed``:
  ``texts`` + required ``search_query``/``search_document`` intent.

The adapter supports both explicitly and never treats an endpoint change as a
mere URL substitution with the old payload shape.
"""

import asyncio
import httpx
import logging
import os
from urllib.parse import urlsplit
from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit

logger = logging.getLogger(__name__)

_NATIVE_INPUT_TYPES = {
    "query": "search_query",
    "document": "search_document",
}


def _embedding_protocol(url: str) -> str:
    configured = (os.getenv("CLOUD_EMBEDDING_PROTOCOL") or "").strip().lower()
    aliases = {
        "cohere": "cohere_native",
        "cohere_v1": "cohere_native",
        "cohere_v2": "cohere_native",
        "native": "cohere_native",
        "foundry": "foundry_models",
        "foundry_models": "foundry_models",
        "generic": "foundry_models",
    }
    if configured:
        protocol = aliases.get(configured, configured)
        if protocol not in {"cohere_native", "foundry_models"}:
            raise ValueError(f"Unsupported CLOUD_EMBEDDING_PROTOCOL={configured}")
        return protocol
    path = urlsplit(url).path.rstrip("/").lower()
    if path.endswith("/v1/embed") or path.endswith("/v2/embed"):
        return "cohere_native"
    return "foundry_models"


def _auth_headers(protocol: str, api_key: Optional[str]) -> dict[str, str]:
    if not api_key:
        return {}
    if protocol != "cohere_native":
        return {"api-key": api_key}

    # Cohere's public/native v2 contract uses Bearer authentication. Some Azure
    # managed-compute deployments expose a raw Authorization key; make that
    # difference explicit rather than guessing based on the hostname.
    scheme = (os.getenv("CLOUD_EMBEDDING_AUTH_SCHEME") or "bearer").strip().lower()
    if scheme in {"bearer", "token"}:
        return {"Authorization": f"Bearer {api_key}"}
    if scheme in {"raw", "authorization"}:
        return {"Authorization": api_key}
    if scheme in {"api-key", "api_key", "apikey"}:
        return {"api-key": api_key}
    raise ValueError(f"Unsupported CLOUD_EMBEDDING_AUTH_SCHEME={scheme}")


class EmbeddingClient:
    """Client for embedding endpoints.

    Full-evaluation API processes set ``CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE=query``
    so every online retrieval vector is a Cohere query embedding. Corpus builders
    pass ``input_type=document`` explicitly. Native Cohere endpoints require one
    of those semantic-search intents and are rejected if it is missing.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if settings.CLOUD_MODE:
            base_url = settings.cloud_embedding_url() or base_url
        self.base_url = base_url
        self.timeout = timeout or settings.EMBED_TIMEOUT_S
        self.model = model or (settings.CLOUD_EMBEDDING if settings.CLOUD_MODE else "default")
        self.api_key = api_key or (settings.AZURE_AI_FOUNDRY_API_KEY if settings.CLOUD_MODE else None)
        self.protocol = _embedding_protocol(self.base_url) if settings.CLOUD_MODE else "local"
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"embedding:{self.protocol}:{self.base_url}:{self.model}"

    def _payload(self, texts: List[str], input_type: Optional[str]) -> dict:
        if self.protocol == "cohere_native":
            native_type = _NATIVE_INPUT_TYPES.get(str(input_type or "").lower())
            if not native_type:
                raise ValueError(
                    "Cohere-native semantic-search embedding requires input_type=document or query; "
                    "publication corpus builders must use document and retrieval must use query"
                )
            return {
                "model": self.model,
                "texts": texts,
                "input_type": native_type,
                "truncate": "NONE",
                "embedding_types": ["float"],
                "output_dimension": settings.active_embedding_dimension(),
            }

        payload = {"input": texts, "model": self.model}
        if settings.CLOUD_MODE:
            if input_type:
                payload["input_type"] = input_type
            if settings.CLOUD_EMBEDDING_DIMENSION or self.model == "embed-v-4-0":
                payload["dimensions"] = settings.active_embedding_dimension()
        return payload

    async def embed(
        self,
        texts: List[str],
        request_id: Optional[str] = None,
        input_type: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate embeddings for texts.

        Args:
            texts: Texts to embed.
            request_id: Optional trace identifier.
            input_type: Semantic search intent. Use ``document`` for corpus
                chunks and ``query`` for online ANN lookup. ``text`` remains
                supported only by the generic Foundry embeddings contract.
        """
        if not texts:
            return []
        if settings.CLOUD_MODE and input_type is None:
            configured_default = (settings.CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE or "").strip().lower()
            input_type = configured_default or None
        if input_type not in {None, "query", "document", "text"}:
            raise ValueError("input_type must be query, document, text, or None")

        payload = self._payload(texts, input_type)
        headers = _auth_headers(self.protocol, self.api_key)
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            logger.debug(
                "Calling embedding service at %s for %s texts (protocol=%s input_type=%s)",
                self.base_url,
                len(texts),
                self.protocol,
                input_type,
            )
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            embeddings = _parse_embedding_response(response.json())
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding service returned {len(embeddings)} vectors for {len(texts)} inputs"
                )
            expected_dim = settings.active_embedding_dimension() if settings.CLOUD_MODE else None
            if expected_dim and any(len(vector) != expected_dim for vector in embeddings):
                dimensions = sorted({len(vector) for vector in embeddings})
                raise RuntimeError(
                    f"Embedding service returned dimensions {dimensions}; expected {expected_dim}"
                )
            return embeddings
        except Exception as exc:
            if settings.CLOUD_MODE:
                logger.error("Embedding request failed: %s", exc)
                raise
            local = await self._local_embed(texts)
            if local is not None:
                logger.warning(
                    "HTTP embedding at %s failed (%s); using locally loaded ModelManager",
                    self.base_url,
                    exc,
                )
                return local
            logger.error("Embedding request failed: %s", exc)
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
        await self.client.aclose()


def _parse_embedding_response(data) -> List[np.ndarray]:
    """Parse generic Foundry, Cohere v1, and Cohere v2 float responses."""
    embeddings: List[np.ndarray] = []
    if isinstance(data, dict) and "data" in data:
        for item in data["data"]:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(np.array(item["embedding"], dtype=np.float32))
        return embeddings

    if isinstance(data, dict) and "embeddings" in data:
        raw = data["embeddings"]
        if isinstance(raw, dict):
            # Cohere v2 returns embeddings by requested type. Some SDK/model
            # serializations spell the float field ``float_``.
            raw = raw.get("float") or raw.get("float_") or []
        if isinstance(raw, list):
            return [np.array(emb, dtype=np.float32) for emb in raw]

    if isinstance(data, list):
        return [np.array(emb, dtype=np.float32) for emb in data]
    return embeddings


def _mean_pool(token_embeddings, attention_mask):
    import torch

    expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
