"""LLM client adapter with Foundry/local-MedSwin generation profiles."""

import httpx
import logging
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit
from app.services.prompts.structured import schema_instruction

logger = logging.getLogger(__name__)

MEDSWIN_MODEL_ID = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"


def _generation_backend() -> str:
    value = (os.getenv("GENERATION_BACKEND") or "").strip().lower()
    if value:
        aliases = {
            "medswin": "medswin_local",
            "local": "medswin_local",
            "local_medswin": "medswin_local",
            "azure": "foundry",
            "cloud": "foundry",
        }
        return aliases.get(value, value)
    return "foundry" if settings.CLOUD_MODE else "local_http"


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints.

    Retrieval can remain in CLOUD_MODE while generation is switched between the
    local MedSwin 7B server and GPT-5.4 in Azure Foundry. This decoupling is
    required for a fair generator ablation on the same corpus/index.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.backend = _generation_backend()
        if self.backend == "medswin_local":
            base_url = os.getenv("MEDSWIN_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
            model = os.getenv("MEDSWIN_LLM_MODEL", MEDSWIN_MODEL_ID)
            api_key = None
        elif self.backend == "foundry":
            # The user's deployment name is surfaced as FOUNDRY_MODEL; retain
            # CLOUD_MODEL as a backward-compatible alias.
            model = os.getenv("FOUNDRY_MODEL") or model or settings.CLOUD_MODEL
            api_key = api_key or settings.AZURE_AI_FOUNDRY_API_KEY
            if settings.CLOUD_MODE:
                base_url = settings.cloud_chat_url()

        self.base_url = base_url
        self.timeout = timeout or settings.LLM_TIMEOUT_S
        self.model = model or (settings.CLOUD_MODEL if settings.CLOUD_MODE else "default")
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"llm:{self.backend}:{self.base_url}:{self.model}"

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = temperature

        if max_tokens:
            if self.model.lower().startswith("gpt-5"):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        if json_schema:
            system_msg = {"role": "system", "content": schema_instruction(json_schema)}
            payload["messages"] = [system_msg] + messages

        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            logger.debug(
                "Calling %s LLM at %s model=%s with %s messages",
                self.backend,
                self.base_url,
                self.model,
                len(messages),
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
            data = response.json()

            content = ""
            token_count = None
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            if "usage" in data:
                token_count = data["usage"].get("total_tokens")
            if not content:
                raise RuntimeError("LLM endpoint returned no completion content")
            return {"content": content, "token_count": token_count, "raw_response": data}
        except httpx.HTTPError as exc:
            logger.error("LLM request failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error in LLM call: %s", exc)
            raise

    async def close(self):
        await self.client.aclose()
