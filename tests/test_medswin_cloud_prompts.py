import inspect
import json

import pytest

from app.core.config import Settings, settings
from app.models.medswin import CandidatePassage, EvidenceBundle, SourceType
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient
from app.services.adapters.reranker import RerankerClient
from app.services.medswin.orchestrator import MedSwinOrchestrator
from app.services.storage import StorageService
from app.services.prompts import answer, emr, guideline, query, safety


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
        return None

    def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_llm_client_injects_schema_once_and_uses_cloud_headers():
    seen = {}
    client = LLMClient("https://example.test/openai/v1/chat/completions", model="gpt-5.4", api_key="key")

    async def fake_post(url, json, headers):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        return FakeResponse({"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    client.client.post = fake_post

    response = await client.call_llm(
        [{"role": "system", "content": query.SYSTEM}, {"role": "user", "content": "q"}],
        json_schema=query.SCHEMA,
        max_tokens=32,
    )

    assert response["content"] == "{\"ok\": true}"
    assert seen["json"]["model"] == "gpt-5.4"
    assert "max_tokens" not in seen["json"]
    assert seen["json"]["max_completion_tokens"] == 32
    assert "temperature" not in seen["json"]
    assert seen["headers"]["api-key"] == "key"
    schema_messages = [
        item for item in seen["json"]["messages"]
        if "Return valid JSON matching this schema" in item["content"]
    ]
    assert len(schema_messages) == 1


@pytest.mark.asyncio
async def test_cloud_embedding_payload_uses_deployment_model_and_api_key():
    seen = {}
    client = EmbeddingClient("https://example.test/openai/v1/embeddings", model="embed-v-4-0", api_key="key")

    async def fake_post(url, json, headers):
        seen["json"] = json
        seen["headers"] = headers
        return FakeResponse({"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    client.client.post = fake_post

    embeddings = await client.embed(["sample"])

    assert seen["json"] == {"input": ["sample"], "model": "embed-v-4-0"}
    assert seen["headers"]["api-key"] == "key"
    assert embeddings[0].tolist() == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_embedding_client_retries_after_rate_limit_cooldown(monkeypatch):
    calls = 0
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_BASE_COOLDOWN_S", 0.0)
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_MAX_COOLDOWN_S", 0.0)
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_JITTER_S", 0.0)

    client = EmbeddingClient("https://example.test/openai/v1/embeddings", model="embed-v-4-0", api_key="key")

    async def fake_post(url, json, headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse({"error": "rate limited"}, status_code=429, headers={"Retry-After": "0"})
        return FakeResponse({"data": [{"embedding": [0.4, 0.5]}]})

    client.client.post = fake_post

    embeddings = await client.embed(["sample"])

    assert calls == 2
    assert embeddings[0].tolist() == pytest.approx([0.4, 0.5])


@pytest.mark.asyncio
async def test_cohere_reranker_response_maps_relevance_scores():
    seen = {}
    client = RerankerClient("https://example.test/providers/cohere/v2/rerank", model="Cohere-rerank-v4.0-fast", api_key="key", provider="cohere")

    async def fake_post(url, json, headers):
        seen["json"] = json
        seen["headers"] = headers
        return FakeResponse({"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.42}]})

    client.client.post = fake_post

    results = await client.rerank("query", ["a", "b"])

    assert seen["json"]["model"] == "Cohere-rerank-v4.0-fast"
    assert seen["json"]["documents"] == ["a", "b"]
    assert seen["headers"]["api-key"] == "key"
    assert results[0]["index"] == 1
    assert results[0]["p_hat"] == 0.91
    assert results[0]["calibration_version"] == "identity:cohere-v2"


def test_prompt_modules_have_direct_system_prompts_and_schemas():
    for module in [answer, emr, guideline, query, safety]:
        assert module.SYSTEM.strip()
        assert "Return JSON only" in module.SYSTEM
        assert module.SCHEMA["type"] == "object"


def test_orchestrator_no_longer_contains_inline_you_are_prompts():
    source = inspect.getsource(MedSwinOrchestrator)
    assert "You are a " not in source
    assert "json_schema={" not in source


def test_settings_accept_azure_and_cloud_defaults_without_env_file():
    config = Settings(
        _env_file=None,
        AZURE_AI_FOUNDRY_ENDPOINT="https://example.services.ai.azure.com",
        AZURE_AI_FOUNDRY_API_KEY="secret",
    )

    assert config.CLOUD_MODE is False
    assert config.CLOUD_MODEL == "gpt-5.4"
    assert config.CLOUD_EMBEDDING == "embed-v-4-0"
    assert config.cloud_chat_url() == "https://example.services.ai.azure.com/openai/v1/chat/completions"


def test_storage_filters_stale_cloud_embedding_space(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setattr(settings, "CLOUD_EMBEDDING", "embed-v-4-0")
    monkeypatch.setattr(settings, "CLOUD_EMBEDDING_DIMENSION", 1536)

    service = StorageService()
    stale = service._stale_embedding_filter(org_id="bench-org")
    active = service._index_embedding_filter()

    assert {"embedding_space": {"$ne": "cloud:embed-v-4-0"}} in stale["$or"]
    assert {"embedding_dim": {"$ne": 1536}} in stale["$or"]
    assert active["embedding_model"] == "embed-v-4-0"
    assert active["embedding_space"] == "cloud:embed-v-4-0"
    assert active["embedding_dim"] == 1536
    assert stale["org_id"] == "bench-org"


def test_final_answer_renderer_rejects_fabricated_citations():
    orchestrator = MedSwinOrchestrator(embedding_client=None, reranker_client=None)
    bundle = EvidenceBundle(
        passages=[
            CandidatePassage(
                chunk_id="real",
                doc_id="d1",
                source_type=SourceType.CPG,
                text="Guideline text.",
            )
        ],
        total_tokens=10,
        cpg_count=1,
        emr_count=0,
        lit_count=0,
    )

    rendered = orchestrator._render_final_answer(
        {
            "answer": "Evidence supports clinician review.",
            "evidence_used": [{"chunk_id": "fake", "use": "not real"}, {"chunk_id": "real", "use": "guideline"}],
            "uncertainty": "Low for the cited guideline only.",
            "contraindications_risks": [],
            "next_steps": [],
            "insufficient_evidence": False,
        },
        bundle,
    )

    assert "real: guideline" in rendered
    assert "fake" not in rendered
