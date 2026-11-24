"""Bedrock request contracts. These tests do not call AWS."""

import pytest

from app.core.config import Settings, settings
from app.models.download import ModelDownloadService
from app.services.adapters.bedrock import (
    converse_request,
    embed_request,
    parse_converse_response,
    parse_embed_response,
    parse_rerank_response,
    rerank_request,
)
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient
from app.services.adapters.reranker import RerankerClient


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def converse(self, model_id, request):
        self.calls.append(("converse", model_id, request))
        return {
            "output": {"message": {"content": [{"text": "MEDSWIN_BEDROCK_OK"}]}},
            "usage": {"totalTokens": 12},
        }

    def invoke_model(self, model_id, body):
        self.calls.append(("invoke", model_id, body))
        if "texts" in body:
            width = int(body["output_dimension"])
            row = [0.0] * width
            row[0] = 0.25
            return {"embeddings": {"float": [row for _ in body["texts"]]}}
        return {
            "results": [
                {"index": 1, "relevance_score": 0.2},
                {"index": 0, "relevance_score": 0.9},
            ]
        }


def test_embed_request_uses_cohere_roles_and_refuses_a_bad_dimension():
    body = embed_request(["aspirin"], "query", 1536)
    assert body["input_type"] == "search_query"
    assert body["output_dimension"] == 1536
    document = embed_request(["note"], "document", 1024)
    assert document["input_type"] == "search_document"
    with pytest.raises(ValueError):
        embed_request(["note"], "query", 768)


def test_parsers_reject_a_short_batch_and_rank_rerank_scores():
    vectors = parse_embed_response({"embeddings": {"float": [[1.0, 0.0]]}}, 1, 2)
    assert vectors == [[1.0, 0.0]]
    with pytest.raises(RuntimeError):
        parse_embed_response({"embeddings": {"float": [[1.0]]}}, 1, 2)
    ranked = parse_rerank_response(
        {"results": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.9}]},
        2,
    )
    assert [item["index"] for item in ranked] == [0, 1]
    assert ranked[0]["calibration_version"] == "identity:cohere-rerank-v3-5"
    request = rerank_request("dose", ["a", "b"])
    assert request["top_n"] == 2
    assert request["api_version"] == 2


def test_converse_keeps_system_text_in_the_prefix():
    request = converse_request(
        [
            {"role": "system", "content": "Cite the record."},
            {"role": "user", "content": "What is the dose?"},
        ],
        max_tokens=32,
        temperature=0.0,
    )
    assert request["system"] == [{"text": "Cite the record."}]
    assert request["messages"][0]["role"] == "user"
    assert request["inferenceConfig"]["maxTokens"] == 32
    parsed = parse_converse_response(
        {"output": {"message": {"content": [{"text": "40 mg"}]}}, "usage": {"totalTokens": 4}}
    )
    assert parsed["content"] == "40 mg"
    assert parsed["token_count"] == 4


@pytest.mark.asyncio
async def test_llm_client_calls_converse(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    monkeypatch.setattr(settings, "BEDROCK_CHAT_MODEL", "global.openai.gpt-6-luna")
    monkeypatch.setenv("GENERATION_BACKEND", "bedrock")
    client = LLMClient("http://ignored", model="ignored")
    runtime = FakeRuntime()
    client.bedrock_runtime = runtime
    result = await client.call_llm(
        [{"role": "user", "content": "Reply with exactly MEDSWIN_BEDROCK_OK"}],
        max_tokens=16,
        temperature=0.0,
    )
    assert result["content"] == "MEDSWIN_BEDROCK_OK"
    assert runtime.calls[0][0] == "converse"
    assert runtime.calls[0][1] == "global.openai.gpt-6-luna"
    assert runtime.calls[0][2]["messages"][0]["content"][0]["text"].endswith("MEDSWIN_BEDROCK_OK")
    assert "temperature" not in runtime.calls[0][2]["inferenceConfig"]


@pytest.mark.asyncio
async def test_bedrock_chat_caps_output_when_the_caller_omits_max_tokens(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    monkeypatch.setattr(settings, "BEDROCK_CHAT_MODEL", "global.openai.gpt-6-luna")
    monkeypatch.setattr(settings, "BEDROCK_MAX_OUTPUT_TOKENS", 1024)
    monkeypatch.setenv("GENERATION_BACKEND", "bedrock")
    monkeypatch.delenv("LLM_DEFAULT_MAX_TOKENS", raising=False)
    client = LLMClient("http://ignored", model="ignored")
    runtime = FakeRuntime()
    client.bedrock_runtime = runtime
    await client.call_llm([{"role": "user", "content": "What is the dose?"}])
    assert runtime.calls[0][2]["inferenceConfig"]["maxTokens"] == 1024


@pytest.mark.asyncio
async def test_embedding_and_rerank_clients_call_invoke_model(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    runtime = FakeRuntime()
    embedding = EmbeddingClient("http://ignored")
    embedding.bedrock_runtime = runtime
    vectors = await embedding.embed(["aspirin"], input_type="document")
    assert len(vectors[0]) == 1536
    assert float(vectors[0][0]) == pytest.approx(0.25)
    assert runtime.calls[0][2]["input_type"] == "search_document"
    assert runtime.calls[0][1] == "global.cohere.embed-v4:0"

    reranker = RerankerClient("http://ignored")
    reranker.bedrock_runtime = runtime
    ranked = await reranker.rerank("aspirin", ["relevant", "unrelated"])
    assert ranked[0]["index"] == 0
    assert ranked[0]["score"] == 0.9
    assert runtime.calls[1][1] == "cohere.rerank-v3-5:0"


def test_download_catalog_uses_the_named_local_models():
    service = ModelDownloadService()
    assert service.models["embedding"]["repo_id"] == "abhinand/MedEmbed-large-v0.1"
    assert service.models["reranker"]["repo_id"] == "BAAI/bge-reranker-v2-m3"


def test_cli_process_credentials_reads_a_login_payload(monkeypatch):
    class Proc:
        returncode = 0
        stdout = (
            '{"Version":1,"AccessKeyId":"AKIATEST","SecretAccessKey":"secret-value",'
            '"SessionToken":"session-token"}'
        )
        stderr = ""

    monkeypatch.setattr("app.services.adapters.bedrock.shutil.which", lambda name: "/usr/bin/aws")
    monkeypatch.setattr("app.services.adapters.bedrock.subprocess.run", lambda *args, **kwargs: Proc())
    from app.services.adapters.bedrock import cli_process_credentials

    creds = cli_process_credentials()
    assert creds["aws_access_key_id"] == "AKIATEST"
    assert creds["aws_session_token"] == "session-token"
    assert "Secret" not in creds


def test_runtime_uses_cli_login_when_the_default_chain_is_empty(monkeypatch):
    created = {}

    class Session:
        def get_credentials(self):
            return None

    class Boto:
        def Session(self):
            return Session()

        def client(self, name, **kwargs):
            created["name"] = name
            created["region"] = kwargs.get("region_name")
            created["has_key"] = bool(kwargs.get("aws_access_key_id"))
            return object()

    monkeypatch.setattr("app.services.adapters.bedrock._import_boto3", lambda: Boto())
    monkeypatch.setattr(
        "app.services.adapters.bedrock.cli_process_credentials",
        lambda: {"aws_access_key_id": "AKIATEST", "aws_secret_access_key": "secret-value"},
    )
    from app.services.adapters.bedrock import BedrockRuntime

    BedrockRuntime("ap-southeast-2").client()
    assert created == {"name": "bedrock-runtime", "region": "ap-southeast-2", "has_key": True}


def test_foundry_backend_follows_the_bedrock_provider():
    config = Settings(_env_file=None, CLOUD_MODE=True, CLOUD_PROVIDER="bedrock", GENERATION_BACKEND="foundry")
    assert config.resolved_generation_backend() == "bedrock"
    assert config.active_generation_model() == "global.openai.gpt-6-luna"
    assert config.active_embedding_model() == "global.cohere.embed-v4:0"
    assert config.active_reranker_model() == "cohere.rerank-v3-5:0"
    assert config.active_embedding_dimension() == 1536
