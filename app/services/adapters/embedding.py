"""Embedding client adapter with retries, provider-aware payloads, and local fallback."""

import asyncio
import httpx
import logging
from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for embedding endpoints.

    Full-evaluation API processes set ``CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE=query``
    so every online retrieval vector is a Cohere query embedding. Corpus builders
    must pass ``input_type=document`` explicitly. Ordinary deployments that do
    not opt into this default preserve the provider's generic/text behavior.
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
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"embedding:{self.base_url}:{self.model}"

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
            input_type: Foundry embedding input type: ``query``, ``document``,
                or ``text``. Corpus ingestion should use ``document`` and ANN
                lookup should use ``query``.
        """
        if not texts:
            return []
        if settings.CLOUD_MODE and input_type is None:
            configured_default = (settings.CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE or "").strip().lower()
            input_type = configured_default or None
        if input_type not in {None, "query", "document", "text"}:
            raise ValueError("input_type must be query, document, text, or None")

        payload = {"input": texts, "model": self.model}
        if settings.CLOUD_MODE:
            if input_type:
                payload["input_type"] = input_type
            if settings.CLOUD_EMBEDDING_DIMENSION or self.model == "embed-v-4-0":
                payload["dimensions"] = settings.active_embedding_dimension()

        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            logger.debug(
                "Calling embedding service at %s for %s texts (input_type=%s)",
                self.base_url,
                len(texts),
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
