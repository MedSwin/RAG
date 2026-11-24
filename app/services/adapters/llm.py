"""LLM client adapter for Bedrock, Foundry, and the local MedSwin server."""

import asyncio
import httpx
import logging
import os
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.adapters.limiter import request_with_model_rate_limit
from app.services.prompts.structured import JSON_SCHEMA_SYSTEM, schema_instruction

logger = logging.getLogger(__name__)

MEDSWIN_MODEL_ID = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"


def _generation_backend() -> str:
    # Process-level override is intentional: the strict matrix swaps generator
    # profiles between API subprocesses while using the same cloud retrieval
    # configuration. Direct uvicorn/.env launches use the first-class setting.
    return settings.resolved_generation_backend()


def _default_max_tokens() -> Optional[int]:
    """Return an optional shared generation cap.

    system-eval sets this value for both generator profiles. This prevents
    Foundry GPT from receiving a larger output allowance than the local MedSwin
    model merely because one backend has a larger context window. Ordinary
    application behavior remains uncapped when the variable is absent/zero.
    Bedrock still applies BEDROCK_MAX_OUTPUT_TOKENS.
    """
    raw = (os.getenv("LLM_DEFAULT_MAX_TOKENS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("LLM_DEFAULT_MAX_TOKENS must be an integer")
    return value if value > 0 else None


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: characters over three, or word pieces."""
    if not text:
        return 0
    word_pieces = 0
    for word in text.split():
        word_pieces += max(1, (len(word) + 3) // 4)
    return max(len(text) // 3, word_pieces)


def _estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    return _estimate_tokens("\n".join(str(message.get("content") or "") for message in messages))


def _clamp_local_completion_tokens(requested: int, prompt_tokens: int) -> int:
    """Keep prompt tokens plus completion inside 1,980. Floor is 96 when it fits."""
    room = 1980 - max(0, prompt_tokens)
    if room >= 96:
        capped = min(requested, room)
        if requested >= 96:
            return max(96, capped)
        return max(1, capped)
    if room > 0:
        return room
    return 1


def _redact_local_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a phone, an email, an MRN, or a labelled date of birth before MedSwin."""
    from app.services.medswin.governance import redact_phi_text

    redacted: List[Dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = redact_phi_text(content)
        redacted.append(item)
    return redacted


def fit_local_messages(messages: List[Dict[str, Any]], completion_tokens: int) -> List[Dict[str, Any]]:
    """Shrink the last user message until the prompt leaves room for the completion."""
    fitted = [dict(message) for message in messages]
    requested = max(1, int(completion_tokens))
    floor = 96 if requested >= 96 else requested
    for budget in (requested, floor, 1):
        if _shrink_user_message(fitted, 1980 - budget):
            return fitted
    return fitted


def _chart_block(content: str, width: int) -> str:
    marker = "Patient chart:\n"
    if marker not in content:
        return ""
    tail = content.split(marker, 1)[1]
    stop = len(tail)
    for header in ("\n\nFacet coverage:", "\n\nEvidence:", "\n\nSafety notes:", "\n\nValid chunk_ids:"):
        index = tail.find(header)
        if index >= 0:
            stop = min(stop, index)
    block = tail[:stop].strip()
    if not block:
        return ""
    return f"{marker}{block[:width]}"


def _evidence_heads(content: str, width: int) -> str:
    if "Evidence:\n" not in content:
        return ""
    evidence = content.split("Evidence:\n", 1)[1]
    for header in ("\n\nValid chunk_ids:", "\n\nSafety notes:"):
        if header in evidence:
            evidence = evidence.split(header, 1)[0]
    blocks = []
    for block in evidence.split("\n\n"):
        text = block.strip()
        if not text:
            continue
        blocks.append(text[:width])
        if len(blocks) >= 4:
            break
    if not blocks:
        return ""
    return "Evidence:\n" + "\n\n".join(blocks)


def _passage_heads(content: str, width: int) -> str:
    """Keep the head of each shared passage. A later caution stays in its own head."""
    marker = "Shared Evidence Passages:\n"
    if marker not in content:
        return ""
    body = content.split(marker, 1)[1]
    for stop in ("\n\nSpecialist Instructions:\n", "\n\nPatient chart:\n", "\n\nQuery:"):
        index = body.find(stop)
        if index >= 0:
            body = body[:index]
    blocks = []
    for block in body.split("\n\n"):
        text = block.strip()
        if not text:
            continue
        blocks.append(text[:width])
        if len(blocks) >= 4:
            break
    if not blocks:
        return ""
    return marker + "\n\n".join(blocks)


def _query_line(content: str) -> str:
    if "Query:" not in content:
        return ""
    rest = content.split("Query:", 1)[1]
    stop = len(rest)
    for header in ("\n\nPatient chart:", "\n\nFacet coverage:", "\n\nEvidence:"):
        index = rest.find(header)
        if index >= 0:
            stop = min(stop, index)
    line = f"Query:{rest[:stop]}".strip()
    return line[:240]


def _compact_user_message(content: str, width: int) -> str:
    """Keep the chart and each passage head. Drop the history tail."""
    chart = _chart_block(content, width)
    evidence = _evidence_heads(content, width)
    passages = _passage_heads(content, width)
    if not chart and not evidence and not passages:
        return content[: max(80, len(content) // 2)]
    if passages:
        question = _query_line(content)
    elif "Patient chart:\n" in content:
        question = content.split("Patient chart:\n", 1)[0].strip()[:240]
    elif "Evidence:\n" in content:
        question = content.split("Evidence:\n", 1)[0].strip()[:240]
    else:
        question = content[:240]
    parts = [part for part in (passages, question, chart, evidence) if part]
    return "\n\n".join(parts)


def _shrink_user_message(messages: List[Dict[str, Any]], prompt_budget: int) -> bool:
    width = 480
    for _ in range(16):
        if _estimate_message_tokens(messages) <= prompt_budget:
            return True
        indexes = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        if not indexes:
            return False
        index = indexes[-1]
        content = str(messages[index].get("content") or "")
        if len(content) <= 80:
            return False
        if "Patient chart:" in content or "Evidence:\n" in content:
            width = max(80, width // 2)
            messages[index]["content"] = _compact_user_message(content, width)
            continue
        messages[index]["content"] = content[: max(80, len(content) // 2)]
    return _estimate_message_tokens(messages) <= prompt_budget


def _schema_already_in_messages(messages: List[Dict[str, Any]], local_budget: bool) -> bool:
    """A second schema block wastes the 2,048-token window."""
    for message in messages:
        if message.get("role") != "system":
            continue
        content = str(message.get("content") or "")
        if JSON_SCHEMA_SYSTEM in content:
            return True
        if local_budget and "Return one JSON object" in content:
            return True
    return False


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints.

    Retrieval can remain in CLOUD_MODE while generation is switched between the
    local MedSwin 7B server and the cloud model. The cloud model is GPT-6 Luna
    on Bedrock, or the Foundry deployment when CLOUD_PROVIDER=foundry.
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
            base_url = os.getenv("MEDSWIN_LLM_URL") or settings.MEDSWIN_LLM_URL
            model = os.getenv("MEDSWIN_LLM_MODEL") or settings.MEDSWIN_LLM_MODEL or MEDSWIN_MODEL_ID
            api_key = None
        elif self.backend == "bedrock":
            base_url = f"bedrock://{settings.AWS_BEDROCK_REGION}"
            model = os.getenv("BEDROCK_CHAT_MODEL") or settings.BEDROCK_CHAT_MODEL or model
            api_key = None
        elif self.backend == "foundry":
            model = os.getenv("FOUNDRY_MODEL") or settings.active_generation_model() or model
            api_key = api_key or settings.AZURE_AI_FOUNDRY_API_KEY
            if settings.CLOUD_MODE:
                base_url = settings.cloud_chat_url()

        self.base_url = base_url
        self.timeout = timeout or settings.LLM_TIMEOUT_S
        self.model = model or (settings.active_generation_model() if settings.CLOUD_MODE else "default")
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
        local_budget = self.backend in {"medswin_local", "local_http"} and (
            not settings.CLOUD_MODE or self.backend == "medswin_local"
        )
        if local_budget:
            messages = _redact_local_messages(messages)
        payload = {"model": self.model, "messages": messages}
        if not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = temperature

        supplied_max_tokens = max_tokens if max_tokens is not None else _default_max_tokens()
        if local_budget and supplied_max_tokens is None:
            supplied_max_tokens = int(os.getenv("MEDSWIN_DEFAULT_MAX_TOKENS", "384"))
        effective_max_tokens = supplied_max_tokens
        if not effective_max_tokens and self.backend == "bedrock":
            # Luna can emit 128000 tokens. A clinical answer stays inside this cap.
            configured = int(settings.BEDROCK_MAX_OUTPUT_TOKENS or 0)
            effective_max_tokens = configured if configured > 0 else 1024

        if json_schema and not _schema_already_in_messages(messages, local_budget):
            system_msg = {"role": "system", "content": schema_instruction(json_schema)}
            payload["messages"] = [system_msg] + messages
        if local_budget and effective_max_tokens:
            payload["messages"] = fit_local_messages(payload["messages"], effective_max_tokens)
            prompt_tokens = _estimate_message_tokens(payload["messages"])
            effective_max_tokens = _clamp_local_completion_tokens(effective_max_tokens, prompt_tokens)
        if effective_max_tokens:
            if self.model.lower().startswith("gpt-5"):
                payload["max_completion_tokens"] = effective_max_tokens
            else:
                payload["max_tokens"] = effective_max_tokens

        if self.backend == "bedrock":
            return await self._call_bedrock(
                payload["messages"],
                max_tokens=effective_max_tokens,
                temperature=temperature,
            )

        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            logger.debug(
                "Calling %s LLM at %s model=%s with %s messages max_tokens=%s",
                self.backend,
                self.base_url,
                self.model,
                len(messages),
                effective_max_tokens,
            )
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers,
                replay_safe=False,
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

    async def _call_bedrock(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: float,
    ) -> Dict[str, Any]:
        from app.services.adapters.bedrock import BedrockRuntime, converse_request, parse_converse_response

        request = converse_request(
            messages,
            max_tokens=max_tokens,
            temperature=None if "gpt-6" in (self.model or "") else temperature,
        )
        runtime = getattr(self, "bedrock_runtime", None) or BedrockRuntime(settings.AWS_BEDROCK_REGION)
        loop = asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(None, lambda: runtime.converse(self.model, request))
        except Exception as exc:
            logger.error("Bedrock Converse failed: %s", type(exc).__name__)
            raise
        return parse_converse_response(payload)

    async def close(self):
        await self.client.aclose()
