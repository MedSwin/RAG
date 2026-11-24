"""Tests for scripts/serve-medswin-llm.py runtime fixes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "serve-medswin-llm.py"
MODEL_DIR = ROOT / "models" / "MedSwin-DaRE-TIES-KD-0.7"


def _import_serve_module():
    name = "serve_medswin_llm"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_medswin_prompt_drops_a_phone_number():
    serve = _import_serve_module()

    class _Tokenizer:
        chat_template = None

    prompt = serve._render_prompt(
        _Tokenizer(),
        [{"role": "user", "content": "Call 555-234-5678. Creatinine is 1.4. Continue metformin."}],
    )
    assert "555-234-5678" not in prompt
    assert "[REDACTED_PHONE]" in prompt
    assert "Creatinine is 1.4." in prompt
    assert "metformin" in prompt


@pytest.fixture
def serve():
    mod = _import_serve_module()
    mod._tokenizer = None
    mod._model = None
    mod._device = None
    mod.reset_prefix_cache()
    yield mod
    mod._tokenizer = None
    mod._model = None
    mod._device = None
    mod.reset_prefix_cache()


class TestContextWindow:
    def test_prefers_model_2048_over_tokenizer_512(self, serve):
        model = SimpleNamespace(
            config=SimpleNamespace(
                max_position_embeddings=2048,
                max_sequence_length=2048,
                n_positions=None,
            )
        )
        tokenizer = SimpleNamespace(model_max_length=512)
        assert serve._context_window(model, tokenizer) == 2048

    def test_falls_back_to_tokenizer_when_model_has_no_limits(self, serve):
        model = SimpleNamespace(config=SimpleNamespace())
        tokenizer = SimpleNamespace(model_max_length=1024)
        assert serve._context_window(model, tokenizer) == 1024

    def test_env_override_clamped_to_model_limit(self, serve, monkeypatch):
        monkeypatch.setenv("MEDSWIN_CONTEXT_WINDOW", "4096")
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=2048, max_sequence_length=None)
        )
        tokenizer = SimpleNamespace(model_max_length=512)
        assert serve._context_window(model, tokenizer) == 2048

    def test_env_override_without_model_limit(self, serve, monkeypatch):
        monkeypatch.delenv("MEDSWIN_CONTEXT_WINDOW", raising=False)
        monkeypatch.setenv("MEDSWIN_CONTEXT_WINDOW", "768")
        model = SimpleNamespace(config=SimpleNamespace())
        tokenizer = SimpleNamespace(model_max_length=512)
        assert serve._context_window(model, tokenizer) == 768


class TestSnapshotRevision:
    def test_marker_when_present(self, serve, monkeypatch, tmp_path):
        monkeypatch.setattr(serve, "MODEL_PATH", tmp_path)
        marker = {
            "model_id": serve.MODEL_ID,
            "revision": "abc123def456",
        }
        (tmp_path / ".medswin_snapshot.json").write_text(json.dumps(marker), encoding="utf-8")
        assert serve._marker_revision() == "abc123def456"
        assert serve._snapshot_revision() == "abc123def456"

    def test_local_hash_when_marker_missing(self, serve, monkeypatch, tmp_path):
        monkeypatch.setattr(serve, "MODEL_PATH", tmp_path)
        (tmp_path / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}', encoding="utf-8")
        (tmp_path / "model-00001.safetensors").write_bytes(b"fake")
        rev = serve._local_safetensors_revision()
        assert rev.startswith("local-safetensors:")
        assert len(rev.split(":")[1]) == 12

    def test_require_snapshot_marker_raises(self, serve, monkeypatch, tmp_path):
        monkeypatch.setattr(serve, "MODEL_PATH", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("MEDSWIN_REQUIRE_SNAPSHOT_MARKER", "true")
        with pytest.raises(RuntimeError, match="snapshot revision marker"):
            serve._load()

    def test_load_succeeds_without_marker_when_config_present(self, serve, monkeypatch, tmp_path):
        tiny = _tiny_llama_dir(tmp_path / "tiny-model")
        monkeypatch.setattr(serve, "MODEL_PATH", tiny)
        monkeypatch.delenv("MEDSWIN_REQUIRE_SNAPSHOT_MARKER", raising=False)
        monkeypatch.setenv("MEDSWIN_DEVICE", "cpu")

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=256,
            pad_token_id=0,
            eos_token_id=1,
        )
        stub_tokenizer = MagicMock()
        monkeypatch.setattr(
            serve,
            "AutoTokenizer",
            MagicMock(from_pretrained=lambda *_a, **_k: stub_tokenizer),
        )
        monkeypatch.setattr(
            serve,
            "_from_pretrained_with_dtype",
            lambda *_a, **_k: LlamaForCausalLM(config),
        )
        serve._load()
        assert serve._snapshot_revision().startswith("local-safetensors:")


def _tiny_llama_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    return path


class TestDeviceDtypeSelection:
    def test_cpu_default_float32(self, serve, monkeypatch):
        monkeypatch.delenv("MEDSWIN_DEVICE", raising=False)
        monkeypatch.delenv("MEDSWIN_DTYPE", raising=False)
        device = serve._select_device()
        if not torch.cuda.is_available() and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            assert device.type == "cpu"
            assert serve._select_dtype(device) == torch.float32

    def test_dtype_override(self, serve, monkeypatch):
        monkeypatch.setenv("MEDSWIN_DTYPE", "float16")
        assert serve._select_dtype(torch.device("cpu")) == torch.float16


class TestTokenizerAndTokenIds:
    @pytest.mark.skipif(not MODEL_DIR.exists(), reason="MedSwin weights not present")
    def test_real_tokenizer_encodes_non_empty_prompt(self, serve):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        prompt = serve._render_prompt(tokenizer, [{"role": "user", "content": "Brief clinical question."}])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        assert int(encoded["input_ids"].shape[-1]) > 0

    def test_resolve_token_ids_prefers_model_config(self, serve):
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2, pad_token_id=32000))
        tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=None)
        pad_id, eos_id = serve._resolve_token_ids(model, tokenizer)
        assert eos_id == 2
        assert pad_id == 32000


class TestHttpValidation:
    def test_default_max_tokens_not_422_with_2048_window(self, serve, monkeypatch):
        tiny = _tiny_llama_in_memory()
        monkeypatch.setattr(serve, "_load", lambda: (tiny["tokenizer"], tiny["model"], torch.device("cpu")))
        monkeypatch.setattr(serve, "_context_window", lambda model, tokenizer: 2048)
        monkeypatch.setenv("MEDSWIN_ENABLE_PREFIX_CACHE", "0")
        with TestClient(serve.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "medswin",
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["context_window"] == 2048
        assert body["usage"]["completion_tokens"] >= 1

    def test_422_when_completion_budget_equals_context_window(self, serve, monkeypatch):
        tiny = _tiny_llama_in_memory()
        monkeypatch.setattr(serve, "_load", lambda: (tiny["tokenizer"], tiny["model"], torch.device("cpu")))
        monkeypatch.setattr(serve, "_context_window", lambda model, tokenizer: 128)
        with TestClient(serve.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 128,
                },
            )
        assert response.status_code == 422

    def test_422_on_prompt_overflow(self, serve, monkeypatch):
        tiny = _tiny_llama_in_memory()
        tiny["tokenizer"]._ids = torch.ones(1, 40, dtype=torch.long)
        monkeypatch.setattr(serve, "_load", lambda: (tiny["tokenizer"], tiny["model"], torch.device("cpu")))
        monkeypatch.setattr(serve, "_context_window", lambda model, tokenizer: 32)
        long_text = "word " * 200
        with TestClient(serve.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "medswin_local",
                    "messages": [{"role": "user", "content": long_text}],
                    "max_tokens": 4,
                },
            )
        assert response.status_code == 422

    def test_accepts_model_aliases(self, serve, monkeypatch):
        tiny = _tiny_llama_in_memory()
        monkeypatch.setattr(serve, "_load", lambda: (tiny["tokenizer"], tiny["model"], torch.device("cpu")))
        with TestClient(serve.app) as client:
            for alias in ("default", "medswin", "medswin_local"):
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": alias, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 4},
                )
                assert response.status_code == 200, alias


class _FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 512

    def __init__(self) -> None:
        self._ids = torch.randint(1, 100, (1, 12))

    def __call__(self, prompt: str, return_tensors: str = "pt", truncation: bool = False):
        _ = prompt, return_tensors, truncation
        return {"input_ids": self._ids.clone(), "attention_mask": torch.ones_like(self._ids)}

    def decode(self, ids, skip_special_tokens: bool = True):
        _ = ids, skip_special_tokens
        return "mocked completion"


def _tiny_llama_in_memory() -> dict[str, Any]:
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=32001,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        max_position_embeddings=2048,
        max_sequence_length=2048,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    tokenizer = _FakeTokenizer()
    return {"model": model, "tokenizer": tokenizer}


class TestPrefixKvCache:
    def test_dynamic_cache_matches_uncached_greedy(self, serve, monkeypatch):
        monkeypatch.setenv("MEDSWIN_ENABLE_PREFIX_CACHE", "1")
        monkeypatch.setenv("MEDSWIN_PREFIX_CACHE_MIN_TOKENS", "4")
        serve.reset_prefix_cache()
        serve._prefix_kv_cache.min_prefix_tokens = 4

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=256,
            pad_token_id=0,
            eos_token_id=1,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        device = torch.device("cpu")

        torch.manual_seed(7)
        input_ids = torch.randint(1, 200, (1, 24))
        attention_mask = torch.ones_like(input_ids)
        gen = {"max_new_tokens": 6, "do_sample": False, "pad_token_id": 0, "eos_token_id": 1}

        uncached = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen, use_cache=True)
        serve._generate_with_optional_prefix_cache(
            model,
            input_ids,
            attention_mask,
            device,
            gen,
            prompt_tokens=int(input_ids.shape[-1]),
            pad_token_id=0,
            eos_token_id=1,
        )
        hits_before = serve._prefix_kv_cache.hits

        cached_out, reused = serve._generate_with_optional_prefix_cache(
            model,
            input_ids,
            attention_mask,
            device,
            gen,
            prompt_tokens=int(input_ids.shape[-1]),
            pad_token_id=0,
            eos_token_id=1,
        )
        assert reused > 0
        assert serve._prefix_kv_cache.hits == hits_before + 1
        assert torch.equal(uncached, cached_out)

    def test_health_reports_kv_cache_stats(self, serve, monkeypatch):
        tiny = _tiny_llama_in_memory()
        monkeypatch.setattr(serve, "_load", lambda: (tiny["tokenizer"], tiny["model"], torch.device("cpu")))
        serve._prefix_kv_cache.hits = 3
        serve._prefix_kv_cache.misses = 1
        serve._prefix_kv_cache.cached_tokens_reused = 40
        with TestClient(serve.app) as client:
            payload = client.get("/health").json()
        assert payload["kv_cache"]["hits"] == 3
        assert payload["kv_cache"]["cached_tokens_reused"] == 40

    def test_chat_response_cached_tokens_field(self, serve, monkeypatch):
        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=256,
            pad_token_id=0,
            eos_token_id=1,
            max_position_embeddings=512,
        )
        model = LlamaForCausalLM(config)
        model.eval()

        class _FixedTokenizer(_FakeTokenizer):
            def __init__(self) -> None:
                self._ids = torch.randint(1, 200, (1, 40))

            def decode(self, ids, skip_special_tokens=True):
                return "hello"

        tokenizer = _FixedTokenizer()
        tokenizer.model_max_length = 999999

        monkeypatch.setattr(
            serve,
            "_load",
            lambda: (tokenizer, model, torch.device("cpu")),
        )
        monkeypatch.setenv("MEDSWIN_ENABLE_PREFIX_CACHE", "1")
        monkeypatch.setenv("MEDSWIN_PREFIX_CACHE_MIN_TOKENS", "8")
        serve.reset_prefix_cache()
        serve._prefix_kv_cache.min_prefix_tokens = 8

        with TestClient(serve.app) as client:
            first = client.post(
                "/v1/chat/completions",
                json={"model": "medswin", "messages": [{"role": "user", "content": "one"}], "max_tokens": 4},
            )
            assert first.status_code == 200
            second = client.post(
                "/v1/chat/completions",
                json={"model": "medswin", "messages": [{"role": "user", "content": "one"}], "max_tokens": 4},
            )
        assert second.status_code == 200
        usage = second.json()["usage"]
        assert usage["cached_tokens"] > 0
        assert serve._prefix_kv_cache.hits >= 1


def test_nonfinite_sampling_retries_greedy(serve):
    class _Boom:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(bool(kwargs.get("do_sample")))
            if kwargs.get("do_sample"):
                raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")

            class _Out:
                sequences = torch.tensor([[1, 2, 3]])
                past_key_values = None

            return _Out()

    model = _Boom()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    sequences, cached = serve._generate_with_optional_prefix_cache(
        model,
        input_ids,
        torch.ones_like(input_ids),
        torch.device("cpu"),
        {"max_new_tokens": 2, "do_sample": True, "temperature": 0.2, "top_p": 0.95},
        prompt_tokens=5,
        pad_token_id=0,
        eos_token_id=1,
    )
    assert model.calls == [True, False]
    assert cached == 0
    assert int(sequences.shape[-1]) == 3