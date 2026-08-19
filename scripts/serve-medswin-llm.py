#!/usr/bin/env python3
"""Serve MedSwin-DaRE-TIES-KD-0.7 behind a minimal OpenAI-compatible endpoint.

This server exists specifically so the RAG/full-system evaluation can swap only
its generator while keeping the same Cohere Embed v4 corpus/index and Cohere
Rerank v4 retrieval stack. It intentionally implements only the endpoint shape
used by app.services.adapters.llm.LLMClient.

Prompt overflow is fail-closed. The server never silently truncates benchmark
prompts: if prompt tokens + requested completion exceed the model-declared
context window, the request fails and the strict matrix records the case as a
failure. The full-eval launcher therefore applies one conservative prompt/output
envelope to both MedSwin and GPT-5.4.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MEDSWIN_LLM_MODEL", "MedSwin/MedSwin-DaRE-TIES-KD-0.7")
MODEL_PATH = Path(os.getenv("MEDSWIN_MODEL_PATH", "./models/MedSwin-DaRE-TIES-KD-0.7"))

app = FastAPI(title="MedSwin local generation server", version="1.1")
_tokenizer = None
_model = None
_device = None


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict[str, Any]]
    temperature: float | None = 0.2
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = 0.95


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_ID
    choices: list[Choice]
    usage: Usage
    context_window: int
    prompt_truncated: bool = False


def _load() -> tuple[Any, Any, Any]:
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"MedSwin model is missing at {MODEL_PATH}; run scripts/warmup-eval.py first"
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if not torch.cuda.is_available():
        _model = _model.to("cpu")
    _model.eval()
    try:
        _device = next(_model.parameters()).device
    except StopIteration:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _tokenizer, _model, _device


def _context_window(model: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    for name in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        value = getattr(getattr(model, "config", None), name, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 10_000_000:
        candidates.append(tokenizer_max)
    if not candidates:
        raise RuntimeError("Unable to determine MedSwin context window from model/tokenizer configuration")
    # Choose the strictest declared limit; this avoids trusting a tokenizer
    # sentinel larger than the actual causal model position table.
    return min(candidates)


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    cleaned = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in messages
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in cleaned) + "\nASSISTANT:"


@app.on_event("startup")
async def startup() -> None:
    _load()


@app.get("/health")
async def health() -> dict[str, Any]:
    tokenizer, model, device = _load()
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "dtype": str(getattr(model, "dtype", None)),
        "context_window": _context_window(model, tokenizer),
        "prompt_policy": "fail_on_overflow_no_truncation",
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.model not in {MODEL_ID, str(MODEL_PATH), MODEL_PATH.name}:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
    tokenizer, model, device = _load()
    context_window = _context_window(model, tokenizer)
    max_new_tokens = int(request.max_completion_tokens or request.max_tokens or 512)
    max_new_tokens = max(1, min(max_new_tokens, int(os.getenv("MEDSWIN_MAX_NEW_TOKENS", "1024"))))
    if max_new_tokens >= context_window:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Requested completion budget {max_new_tokens} leaves no MedSwin prompt context "
                f"inside context_window={context_window}"
            ),
        )

    prompt = _render_prompt(tokenizer, request.messages)
    # Never pass truncation=True here. Silent benchmark truncation would make the
    # local-vs-cloud comparison impossible to audit.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if prompt_tokens + max_new_tokens > context_window:
        raise HTTPException(
            status_code=422,
            detail=(
                "MedSwin prompt exceeds the declared context window without truncation: "
                f"prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}, "
                f"context_window={context_window}. Reduce the shared full-eval prompt/evidence envelope."
            ),
        )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    temperature = float(request.temperature if request.temperature is not None else 0.2)
    do_sample = temperature > 0.0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = float(request.top_p or 0.95)

    try:
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="MedSwin generation ran out of GPU memory") from exc

    generated_ids = output[0][prompt_tokens:]
    completion_tokens = int(generated_ids.shape[-1])
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not content:
        raise HTTPException(status_code=500, detail="MedSwin generated an empty completion")
    return ChatResponse(
        model=MODEL_ID,
        choices=[Choice(message=ChoiceMessage(content=content))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        context_window=context_window,
        prompt_truncated=False,
    )


if __name__ == "__main__":
    import uvicorn

    # Pass the app object directly. The filename contains hyphens, so using a
    # module import string such as ``scripts.serve-medswin-llm:app`` is invalid.
    uvicorn.run(
        app,
        host=os.getenv("MEDSWIN_LLM_HOST", "127.0.0.1"),
        port=int(os.getenv("MEDSWIN_LLM_PORT", "8000")),
        reload=False,
    )
