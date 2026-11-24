#!/usr/bin/env python3
"""Serve MedSwin-DaRE-TIES-KD-0.7 behind a minimal OpenAI-compatible endpoint.

This server exists specifically so the RAG/full-system evaluation can swap only
its generator while keeping the same Cohere Embed v4 corpus/index and Cohere
Rerank v4 retrieval stack. It intentionally implements only the endpoint shape
used by app.services.adapters.llm.LLMClient.

Prompt overflow is fail-closed. The server never silently truncates benchmark
prompts: if prompt tokens + requested completion exceed the model-declared
context window, the request fails. system-eval therefore applies one conservative
prompt/output envelope to both MedSwin and Foundry GPT.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_ID = os.getenv("MEDSWIN_LLM_MODEL", "MedSwin/MedSwin-DaRE-TIES-KD-0.7")
MODEL_PATH = Path(os.getenv("MEDSWIN_MODEL_PATH", "./models/MedSwin-DaRE-TIES-KD-0.7"))

_ACCEPTED_MODEL_ALIASES = {
    MODEL_ID,
    str(MODEL_PATH),
    MODEL_PATH.name,
    "default",
    "medswin",
    "medswin_local",
}

app = FastAPI(title="MedSwin local generation server", version="1.2")
_tokenizer = None
_model = None
_device: torch.device | None = None


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
    cached_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = MODEL_ID
    choices: list[Choice]
    usage: Usage
    context_window: int
    prompt_truncated: bool = False


def _marker_revision() -> str:
    marker = MODEL_PATH / ".medswin_snapshot.json"
    if not marker.exists():
        return ""
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if payload.get("model_id") != MODEL_ID:
        return ""
    return str(payload.get("revision") or "").strip()


def _local_safetensors_revision() -> str:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        return ""
    config_bytes = config_path.read_bytes()
    index_path = MODEL_PATH / "model.safetensors.index.json"
    if index_path.exists():
        listing = index_path.read_bytes()
    else:
        names = sorted(p.name.encode("utf-8") for p in MODEL_PATH.glob("*.safetensors"))
        listing = b"\n".join(names)
    digest = hashlib.sha256(config_bytes + listing).hexdigest()[:12]
    return f"local-safetensors:{digest}"


def _snapshot_revision() -> str:
    marker = _marker_revision()
    if marker:
        return marker
    return _local_safetensors_revision()


def _require_snapshot_marker() -> bool:
    return os.getenv("MEDSWIN_REQUIRE_SNAPSHOT_MARKER", "").lower() in {"1", "true", "yes"}


def _select_device() -> torch.device:
    override = os.getenv("MEDSWIN_DEVICE", "").strip().lower()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(device: torch.device) -> torch.dtype:
    override = os.getenv("MEDSWIN_DTYPE", "").strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if override in mapping:
        return mapping[override]
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _from_pretrained_with_dtype(path: Path, dtype: torch.dtype, **kwargs: Any) -> Any:
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "out of memory" in message or "mps backend out of memory" in message
    return False


def _load() -> tuple[Any, Any, torch.device]:
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"MedSwin model is missing at {MODEL_PATH}; run scripts/warmup-eval.py first"
        )
    if _require_snapshot_marker() and not _marker_revision():
        raise RuntimeError(
            f"MedSwin snapshot revision marker is missing/invalid at {MODEL_PATH}; "
            "run scripts/warmup-eval.py so the publication model revision is pinned"
        )
    if not _snapshot_revision() and not (MODEL_PATH / "config.json").exists():
        raise RuntimeError(
            f"MedSwin snapshot revision marker is missing/invalid at {MODEL_PATH}; "
            "run scripts/warmup-eval.py so the publication model revision is pinned"
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device = _select_device()
    dtype = _select_dtype(device)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    _model = _from_pretrained_with_dtype(MODEL_PATH, dtype, **kwargs)
    if "device_map" not in kwargs:
        _model = _model.to(device)
    _model.eval()
    try:
        resolved_win = _context_window(_model, _tokenizer)
        if isinstance(getattr(_tokenizer, "model_max_length", None), int) and _tokenizer.model_max_length < resolved_win:
            _tokenizer.model_max_length = resolved_win
    except Exception:
        pass
    try:
        _device = next(_model.parameters()).device
    except StopIteration:
        _device = device
    return _tokenizer, _model, _device


def _model_position_limits(model: Any) -> list[int]:
    limits: list[int] = []
    for name in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        value = getattr(getattr(model, "config", None), name, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            limits.append(value)
    return limits


def _context_window(model: Any, tokenizer: Any) -> int:
    model_limits = _model_position_limits(model)
    if model_limits:
        window = min(model_limits)
    else:
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        if not (isinstance(tokenizer_max, int) and 0 < tokenizer_max < 10_000_000):
            raise RuntimeError(
                "Unable to determine MedSwin context window from model/tokenizer configuration"
            )
        window = tokenizer_max

    raw = os.getenv("MEDSWIN_CONTEXT_WINDOW", "").strip()
    if raw:
        env_val = int(raw)
        if model_limits:
            window = min(env_val, min(model_limits))
        else:
            window = env_val
    return window


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    cleaned = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in messages
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in cleaned) + "\nASSISTANT:"


def _resolve_token_ids(model: Any, tokenizer: Any) -> tuple[int | None, int | None]:
    eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return pad_token_id, eos_token_id


@dataclass
class _PrefixCacheEntry:
    token_ids: tuple[int, ...]
    cache: DynamicCache


@dataclass
class PrefixKvCache:
    max_entries: int = field(
        default_factory=lambda: int(os.getenv("MEDSWIN_PREFIX_CACHE_MAX_ENTRIES", "2"))
    )
    min_prefix_tokens: int = field(
        default_factory=lambda: int(os.getenv("MEDSWIN_PREFIX_CACHE_MIN_TOKENS", "32"))
    )
    hits: int = 0
    misses: int = 0
    cached_tokens_reused: int = 0
    _entries: OrderedDict[str, _PrefixCacheEntry] = field(default_factory=OrderedDict)

    @property
    def enabled(self) -> bool:
        return os.getenv("MEDSWIN_ENABLE_PREFIX_CACHE", "1").lower() not in {"0", "false", "no"}

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.cached_tokens_reused = 0
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "cached_tokens_reused": self.cached_tokens_reused,
        }

    @staticmethod
    def _longest_prefix(token_ids: list[int], entry_ids: tuple[int, ...], max_len: int) -> int:
        limit = min(len(token_ids) - 1, len(entry_ids), max_len)
        matched = 0
        while matched < limit and token_ids[matched] == entry_ids[matched]:
            matched += 1
        return matched

    def lookup(
        self, input_ids: torch.Tensor, device: torch.device
    ) -> tuple[DynamicCache | None, int]:
        if not self.enabled:
            return None, 0
        ids = input_ids[0].tolist()
        if len(ids) < 2:
            return None, 0

        best_len = 0
        best_entry: _PrefixCacheEntry | None = None
        for entry in self._entries.values():
            matched = self._longest_prefix(ids, entry.token_ids, len(entry.token_ids))
            if matched > best_len:
                best_len = matched
                best_entry = entry

        if best_entry is None or best_len < self.min_prefix_tokens:
            self.misses += 1
            return None, 0

        cloned = DynamicCache()
        try:
            for layer_idx, layer in enumerate(best_entry.cache.layers):
                k_src = layer.keys[:, :, :best_len, :]
                v_src = layer.values[:, :, :best_len, :]
                k = (k_src if k_src.device == device else k_src.to(device=device)).clone()
                v = (v_src if v_src.device == device else v_src.to(device=device)).clone()
                cloned.update(k, v, layer_idx=layer_idx)
        except Exception:
            self.misses += 1
            return None, 0

        self.hits += 1
        self.cached_tokens_reused += best_len
        return cloned, best_len

    def store(self, token_ids: list[int], cache: DynamicCache, store_len: int) -> None:
        if not self.enabled or store_len < self.min_prefix_tokens:
            return
        dev_cache = DynamicCache()
        try:
            for layer_idx, layer in enumerate(cache.layers):
                k = layer.keys[:, :, :store_len, :].detach().clone()
                v = layer.values[:, :, :store_len, :].detach().clone()
                dev_cache.update(k, v, layer_idx=layer_idx)
        except Exception:
            return

        key = hashlib.sha256(tuple(token_ids[:store_len]).__repr__().encode()).hexdigest()[:16]
        self._entries[key] = _PrefixCacheEntry(
            token_ids=tuple(token_ids[:store_len]),
            cache=dev_cache,
        )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def touch(self, token_ids: list[int], store_len: int) -> None:
        if not self.enabled:
            return
        key = hashlib.sha256(tuple(token_ids[:store_len]).__repr__().encode()).hexdigest()[:16]
        if key in self._entries:
            self._entries.move_to_end(key)


_prefix_kv_cache = PrefixKvCache()


def reset_prefix_cache() -> None:
    """Clear prefix KV cache state (used by tests)."""
    _prefix_kv_cache.reset()


def _nonfinite_sampling(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "probability tensor" in text or "nan" in text or "inf" in text


def _greedy_generation_kwargs(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    greedy = dict(generation_kwargs)
    greedy["do_sample"] = False
    greedy.pop("temperature", None)
    greedy.pop("top_p", None)
    greedy.pop("top_k", None)
    return greedy


def _generate_with_optional_prefix_cache(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    generation_kwargs: dict[str, Any],
    prompt_tokens: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, int]:
    gen_kwargs = dict(generation_kwargs)
    gen_kwargs["use_cache"] = True
    gen_kwargs["return_dict_in_generate"] = True
    if pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", pad_token_id)
    if eos_token_id is not None:
        gen_kwargs.setdefault("eos_token_id", eos_token_id)

    def _run(kwargs: dict[str, Any], past: Any = None) -> Any:
        call = dict(kwargs)
        if past is not None:
            call["past_key_values"] = past
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **call)

    # MPS float16 prefix reuse can stall inside generate on the second call.
    # CPU reuse stays on. The in-process smoke benchmark still records prefix hits.
    if device.type == "mps":
        prefix_cache, matched = None, 0
    else:
        prefix_cache, matched = _prefix_kv_cache.lookup(input_ids, device)
    if prefix_cache is not None and 0 < matched < prompt_tokens:
        try:
            out = _run(gen_kwargs, prefix_cache)
            _store_prefix_from_generate_output(input_ids, prompt_tokens, out)
            return out.sequences, matched
        except Exception as exc:
            if _nonfinite_sampling(exc):
                gen_kwargs = _greedy_generation_kwargs(gen_kwargs)
            prefix_cache = None

    try:
        out = _run(gen_kwargs)
    except RuntimeError as exc:
        if not _nonfinite_sampling(exc) or not generation_kwargs.get("do_sample"):
            raise
        out = _run(_greedy_generation_kwargs(gen_kwargs))
    _store_prefix_from_generate_output(input_ids, prompt_tokens, out)
    return out.sequences, 0


def _store_prefix_from_generate_output(
    input_ids: torch.Tensor, prompt_tokens: int, out: Any
) -> None:
    store_len = max(prompt_tokens - 1, 0)
    if store_len <= 0:
        return
    pkv = getattr(out, "past_key_values", None)
    if pkv is None:
        return
    try:
        store_cache = DynamicCache()
        for layer_idx, layer in enumerate(pkv.layers):
            store_cache.update(
                layer.keys[:, :, :store_len, :].clone(),
                layer.values[:, :, :store_len, :].clone(),
                layer_idx=layer_idx,
            )
        ids = input_ids[0].tolist()
        _prefix_kv_cache.store(ids, store_cache, store_len)
    except Exception:
        return


@app.on_event("startup")
async def startup() -> None:
    _load()


@app.get("/health")
async def health() -> dict[str, Any]:
    tokenizer, model, device = _load()
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "model_revision": _snapshot_revision(),
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "dtype": str(getattr(model, "dtype", None)),
        "context_window": _context_window(model, tokenizer),
        "prompt_policy": "fail_on_overflow_no_truncation",
        "kv_cache": _prefix_kv_cache.stats(),
    }


_GENERATION_LOCK = threading.Lock()


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.model not in _ACCEPTED_MODEL_ALIASES:
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
                f"context_window={context_window}. Reduce the shared system-eval prompt/evidence envelope."
            ),
        )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    temperature = float(request.temperature if request.temperature is not None else 0.2)
    # MPS float16 multinomial yields non-finite probabilities on long clinical prompts.
    force_greedy = str(device).startswith("mps") and os.getenv("MEDSWIN_MPS_SAMPLE", "0") != "1"
    do_sample = temperature > 0.0 and not force_greedy
    print(
        f"MedSwin generate start prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} sample={do_sample}",
        flush=True,
    )
    pad_token_id, eos_token_id = _resolve_token_ids(model, tokenizer)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = float(request.top_p or 0.95)

    def _run_generate() -> tuple[Any, int]:
        with _GENERATION_LOCK, torch.inference_mode():
            return _generate_with_optional_prefix_cache(
                model,
                input_ids,
                attention_mask,
                device,
                generation_kwargs,
                prompt_tokens,
                pad_token_id,
                eos_token_id,
            )

    try:
        output, cached_tokens = await asyncio.to_thread(_run_generate)
    except Exception as exc:
        if _is_oom_error(exc):
            _empty_device_cache(device)
            raise HTTPException(status_code=503, detail="MedSwin generation ran out of GPU memory") from exc
        raise

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
            cached_tokens=cached_tokens,
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