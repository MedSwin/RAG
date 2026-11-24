"""AWS Bedrock Runtime for chat, Cohere embed, and Cohere rerank.

The model IDs match the Kollab catalog. Chat uses Converse. Embed and rerank
use InvokeModel. Credentials come from the default AWS chain. This module does
not read a static provider key.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Optional

EMBED_DIMENSIONS = {256, 512, 1024, 1536}
MAX_EMBED_BATCH = 96
MAX_RERANK_DOCUMENTS = 100
CHAT_MODEL = "global.openai.gpt-6-luna"
EMBED_MODEL = "global.cohere.embed-v4:0"
RERANK_MODEL = "cohere.rerank-v3-5:0"

_NATIVE_INPUT_TYPES = {
    "query": "search_query",
    "document": "search_document",
}


def embed_request(texts: list[str], input_type: str, dimension: int) -> dict[str, Any]:
    """Build one Cohere Embed v4 InvokeModel body."""
    native = _NATIVE_INPUT_TYPES.get(str(input_type or "").lower())
    if native is None:
        raise ValueError("Bedrock Cohere embed requires input_type=query or document")
    if dimension not in EMBED_DIMENSIONS:
        raise ValueError("Bedrock Cohere embed dimension must be 256, 512, 1024, or 1536")
    if not texts or len(texts) > MAX_EMBED_BATCH:
        raise ValueError(f"Bedrock Cohere embed accepts 1 to {MAX_EMBED_BATCH} texts")
    return {
        "texts": list(texts),
        "input_type": native,
        "embedding_types": ["float"],
        "output_dimension": dimension,
    }


def parse_embed_response(payload: Any, count: int, dimension: int) -> list[list[float]]:
    """Return finite vectors aligned with the request. Refuse a short or drifted batch."""
    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    rows = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
    if not isinstance(rows, list) or len(rows) != count:
        raise RuntimeError("Bedrock embed response does not match the input batch")
    vectors: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != dimension:
            raise RuntimeError("Bedrock embed vector has the wrong length")
        try:
            vector = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Bedrock embed vector is not numeric") from exc
        if any(not math.isfinite(value) for value in vector):
            raise RuntimeError("Bedrock embed vector is not finite")
        vectors.append(vector)
    return vectors


def rerank_request(query: str, documents: list[str]) -> dict[str, Any]:
    """Build one Cohere Rerank 3.5 InvokeModel body."""
    if not query.strip():
        raise ValueError("Bedrock rerank requires a query")
    if not documents or len(documents) > MAX_RERANK_DOCUMENTS:
        raise ValueError(f"Bedrock rerank accepts 1 to {MAX_RERANK_DOCUMENTS} documents")
    return {
        "api_version": 2,
        "query": query,
        "documents": list(documents),
        "top_n": len(documents),
    }


def parse_rerank_response(payload: Any, count: int) -> list[dict[str, Any]]:
    """Map Cohere results onto one score per input index, highest score first."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != count:
        raise RuntimeError("Bedrock rerank response does not cover the input documents")
    covered: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("Bedrock rerank row is malformed")
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if type(index) is not int or not 0 <= index < count or index in covered:
            raise RuntimeError("Bedrock rerank indexes do not cover every document once")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError("Bedrock rerank score is not finite")
        covered[index] = float(score)
    ranked = [
        {
            "index": index,
            "score": covered[index],
            "p_hat": covered[index],
            "calibration_version": "identity:cohere-rerank-v3-5",
        }
        for index in range(count)
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def converse_request(
    messages: list[dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    """Split OpenAI-style messages into a Converse request.

    System turns stay in the prefix so a stable clinical instruction can be
    cached by the service. User and assistant turns stay in order.
    """
    system: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        text = message.get("content")
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if role == "system":
            system.append({"text": text})
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Bedrock Converse does not accept role {role}")
        turns.append({"role": role, "content": [{"text": text}]})
    if not turns or turns[0]["role"] != "user":
        raise ValueError("Bedrock Converse requires the first turn to be a user message")
    inference: dict[str, Any] = {}
    if max_tokens:
        inference["maxTokens"] = int(max_tokens)
    if temperature is not None:
        inference["temperature"] = float(temperature)
    request: dict[str, Any] = {"messages": turns, "inferenceConfig": inference}
    if system:
        request["system"] = system
    return request


def parse_converse_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the assistant text and the total token count."""
    message = ((payload.get("output") or {}).get("message") or {})
    blocks = message.get("content") or []
    text = "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )
    if not text:
        raise RuntimeError("Bedrock Converse returned no text")
    usage = payload.get("usage") or {}
    total = usage.get("totalTokens")
    return {"content": text, "token_count": total, "raw_response": payload}


def _status_of(exc: Exception) -> Optional[int]:
    try:
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError:
        return None
    if isinstance(exc, NoCredentialsError):
        return None
    if isinstance(exc, ClientError):
        meta = exc.response.get("ResponseMetadata") or {}
        return int(meta.get("HTTPStatusCode") or 0) or None
    if isinstance(exc, BotoCoreError):
        return 0
    return None


class BedrockRuntime:
    """Thin IAM client. Tests pass a fake client and do not open a socket."""

    def __init__(self, region: str, client: Any = None, attempts: int = 4):
        self.region = region
        self._client = client
        self.attempts = attempts

    def client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    connect_timeout=5,
                    read_timeout=60,
                    retries={"total_max_attempts": 1, "mode": "standard"},
                ),
            )
        return self._client

    def _call(self, fn):
        delay = 0.4
        last: Optional[Exception] = None
        for attempt in range(self.attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - status decides retry; the body is not logged
                last = exc
                status = _status_of(exc)
                retriable = status in {0, 429, 500, 502, 503, 504}
                if not retriable or attempt >= self.attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
        if last is not None:
            raise last
        raise RuntimeError("Bedrock call failed")

    def invoke_model(self, model_id: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")

        def _once() -> dict[str, Any]:
            response = self.client().invoke_model(
                modelId=model_id,
                body=payload,
                contentType="application/json",
                accept="application/json",
            )
            stream = response["body"]
            try:
                raw = stream.read()
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RuntimeError("Bedrock InvokeModel did not return a JSON object")
            return parsed

        return self._call(_once)

    def converse(self, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": request["messages"],
            "inferenceConfig": request.get("inferenceConfig") or {},
        }
        if request.get("system"):
            kwargs["system"] = request["system"]

        def _once() -> dict[str, Any]:
            response = self.client().converse(**kwargs)
            if not isinstance(response, dict):
                raise RuntimeError("Bedrock Converse did not return an object")
            return response

        return self._call(_once)
