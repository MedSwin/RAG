"""Local three-model adapters, shared specialist prefixes, and 2048-token compaction."""

import io
import json
import urllib.request

import httpx
import numpy as np
import pytest
import torch

from app.agents.base import SPECIALIST_SUFFIX_MARK, passage_context, shared_claim_system_message
from app.agents.critic import ContradictionAgent
from app.agents.emr import EMRAgent
from app.agents.guideline import GuidelineAgent
from app.agents.quality import QualityAgent
from app.agents.safety import SafetyAgent
from app.agents.synthesis import SynthesisAgent
from app.api.v1.endpoints.pipelines import pipelines_ready
from app.core.config import settings
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage, EvidenceBundle
from app.schemas.facets import ClinicalFacet
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient, _estimate_message_tokens, fit_local_messages
from app.services.adapters.reranker import RerankerClient
from benchmarks.system_eval.contracts import GENERATOR_CONTRACTS, resolve_generator_contract
from benchmarks.system_eval.runner import _profile_environment, _retrieval_identity_matches
from benchmarks.trec_cds2016.nist import _download_url
from benchmarks.trec_cds2016.runtime import EXPECTED_LOCAL_MODEL


def _passage(chunk_id, source_type, text, **scores):
    return CandidatePassage(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type=source_type,
        text=text,
        **scores,
    )


class _RecordingLLM:
    def __init__(self, backend="bedrock"):
        self.backend = backend
        self.calls = []

    async def call_llm(self, messages, json_schema=None, temperature=0.7, max_tokens=None, request_id=None):
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return {
            "content": json.dumps(
                {
                    "claims": [
                        {"facet": "safety_contraindications", "claim": "emr fact", "chunk_id": "emr-1", "polarity": "supports"},
                        {"facet": "guideline_concordance", "claim": "cpg fact", "chunk_id": "cpg-1", "polarity": "supports"},
                        {"facet": "safety_contraindications", "claim": "warn", "chunk_id": "safety-1", "polarity": "safety"},
                        {"facet": "clinical_evidence", "claim": "lit fact", "chunk_id": "lit-1", "polarity": "supports"},
                    ],
                    "notes": [],
                }
            )
        }


def _shared_prefix(messages):
    system = messages[0]["content"]
    user = messages[1]["content"]
    head, mark, _suffix = user.partition(SPECIALIST_SUFFIX_MARK)
    assert mark == SPECIALIST_SUFFIX_MARK
    return system, head


def _bundle(passages):
    return EvidenceBundle(
        passages=passages,
        total_tokens=10,
        cpg_count=1,
        emr_count=1,
        lit_count=1,
        safety_count=1,
    )


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_reranker_uses_inprocess_bge_without_http_and_sticks_after_connect_error(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.delenv("LOCAL_INPROCESS_MODELS", raising=False)
    http_calls = {"count": 0}
    scored = {"batches": []}

    class _Reranker:
        def compute_relevance_scores(self, query, passages):
            scored["batches"].append((query, list(passages)))
            return [0.2, 0.9, 0.1]

    class _Manager:
        def get_reranker_model(self):
            return _Reranker()

    monkeypatch.setattr("app.core.state.get_model_manager", lambda: _Manager())

    async def _http(*args, **kwargs):
        http_calls["count"] += 1
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr("app.services.adapters.reranker.request_with_model_rate_limit", _http)
    client = RerankerClient("http://127.0.0.1:8004/rerank")
    first = await client.rerank("pneumonia", ["a", "b", "c"])
    assert http_calls["count"] == 1
    assert client._prefer_inprocess_local is True
    assert first[0]["index"] == 1
    assert first[0]["score"] == pytest.approx(0.9)
    assert first[0]["p_hat"] == pytest.approx(0.9)
    assert first[0]["calibration_version"] == "identity:bge-reranker-v2-m3-local"
    assert [row["index"] for row in first] == [1, 0, 2]

    second = await client.rerank("pneumonia", ["a", "b", "c"])
    assert http_calls["count"] == 1
    assert second[0]["index"] == 1
    assert len(scored["batches"]) == 2


@pytest.mark.asyncio
async def test_reranker_local_url_and_env_skip_http_and_singular_scores_still_work(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setenv("LOCAL_INPROCESS_MODELS", "yes")
    http_calls = {"count": 0}

    class _Singular:
        def compute_relevance_score(self, query, passage):
            return {"low": 0.1, "high": 0.8}[passage]

    class _Manager:
        def get_reranker_model(self):
            return _Singular()

    monkeypatch.setattr("app.core.state.get_model_manager", lambda: _Manager())

    async def _http(*args, **kwargs):
        http_calls["count"] += 1
        raise AssertionError("HTTP reranker should not be called")

    monkeypatch.setattr("app.services.adapters.reranker.request_with_model_rate_limit", _http)
    client = RerankerClient("local://bge-reranker-v2-m3")
    ranked = await client.rerank("q", ["low", "high"])
    assert http_calls["count"] == 0
    assert [row["index"] for row in ranked] == [1, 0]
    assert ranked[0]["calibration_version"] == "identity:bge-reranker-v2-m3-local"


@pytest.mark.asyncio
async def test_the_reranker_sees_a_late_caution(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.delenv("LOCAL_INPROCESS_MODELS", raising=False)
    seen = {}

    async def _http(*args, **kwargs):
        seen["json"] = kwargs["json"]
        return _JsonResponse({"scores": [0.4]})

    monkeypatch.setattr("app.services.adapters.reranker.request_with_model_rate_limit", _http)
    filler = "History note. " * 200
    client = RerankerClient("http://127.0.0.1:8004/rerank")
    await client.rerank("apixaban dose", [f"{filler} Avoid apixaban."])
    document = seen["json"]["passages"][0]
    assert document.startswith("Avoid apixaban.")
    assert "History note." in document


@pytest.mark.asyncio
async def test_reranker_http_success_stays_on_http_and_status_errors_do_not_stick(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.delenv("LOCAL_INPROCESS_MODELS", raising=False)
    calls = {"count": 0}

    async def _http(*args, **kwargs):
        calls["count"] += 1
        return _JsonResponse({"scores": [0.25, 0.75]})

    monkeypatch.setattr("app.services.adapters.reranker.request_with_model_rate_limit", _http)
    client = RerankerClient("http://127.0.0.1:8004/rerank")
    ranked = await client.rerank("q", ["a", "b"])
    assert calls["count"] == 1
    assert ranked[0]["index"] == 1
    assert client._prefer_inprocess_local is False

    request = httpx.Request("POST", "http://127.0.0.1:8004/rerank")
    response = httpx.Response(500, request=request)

    async def _status(*args, **kwargs):
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr("app.services.adapters.reranker.request_with_model_rate_limit", _status)
    with pytest.raises(httpx.HTTPStatusError):
        await client.rerank("q", ["a", "b"])
    assert client._prefer_inprocess_local is False


@pytest.mark.asyncio
async def test_embedding_connect_error_sticks_and_batches_thirty_two(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.delenv("LOCAL_INPROCESS_MODELS", raising=False)
    http_calls = {"count": 0}

    async def _http(*args, **kwargs):
        http_calls["count"] += 1
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.services.adapters.embedding.request_with_model_rate_limit", _http)

    class _Tokenizer:
        def __init__(self):
            self.batches = []

        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            self.batches.append(list(texts))
            return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

    class _Model:
        def __call__(self, **inputs):
            rows = inputs["input_ids"].shape[0]
            hidden = torch.ones((rows, 2, 4))

            class _Out:
                last_hidden_state = hidden

            return _Out()

    tokenizer = _Tokenizer()

    class _Manager:
        def get_embedding_model(self):
            return tokenizer, _Model(), torch.device("cpu"), 4

    monkeypatch.setattr("app.core.state.get_model_manager", lambda: _Manager())
    client = EmbeddingClient("http://127.0.0.1:8005/embeddings")
    texts = [f"passage {index}" for index in range(40)]
    vectors = await client.embed(texts)
    assert http_calls["count"] == 1
    assert client._prefer_inprocess_local is True
    assert tokenizer.batches == [texts[:32], texts[32:]]
    assert len(vectors) == 40
    assert vectors[0].shape == (4,)

    await client.embed(["again"])
    assert http_calls["count"] == 1
    assert tokenizer.batches[-1] == ["again"]


@pytest.mark.asyncio
async def test_the_embedding_sees_a_late_caution(monkeypatch):
    class _Tokenizer:
        def __init__(self):
            self.batches = []

        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            self.batches.append(list(texts))
            ids = torch.ones((len(texts), 2), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    class _Model:
        def __call__(self, **inputs):
            rows = inputs["input_ids"].shape[0]
            hidden = torch.ones((rows, 2, 4))

            class _Out:
                last_hidden_state = hidden

            return _Out()

    tokenizer = _Tokenizer()

    class _Manager:
        def get_embedding_model(self):
            return tokenizer, _Model(), torch.device("cpu"), 4

    monkeypatch.setattr("app.core.state.get_model_manager", lambda: _Manager())
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    client = EmbeddingClient("local://medembed")
    filler = "History note. " * 200
    text = f"{filler} Avoid apixaban."

    await client.embed([text])
    sent = tokenizer.batches[0][0]
    assert sent.startswith("Avoid apixaban.")
    assert "History note." in sent

    from app.services.ingestion import IngestionPipelineService

    IngestionPipelineService._generate_embeddings_sync(
        object(),
        [{"content": text, "chunk_id": "c1"}],
        tokenizer,
        _Model(),
        torch.device("cpu"),
        4,
    )
    ingested = tokenizer.batches[1][0]
    assert ingested.startswith("Avoid apixaban.")
    assert "History note." in ingested


@pytest.mark.asyncio
async def test_the_local_embedding_drops_a_phone_number(monkeypatch):
    class _Tokenizer:
        def __init__(self):
            self.batches = []

        def __call__(self, texts, truncation, padding, max_length, return_tensors):
            self.batches.append(list(texts))
            ids = torch.ones((len(texts), 2), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    class _Model:
        def __call__(self, **inputs):
            rows = inputs["input_ids"].shape[0]
            hidden = torch.ones((rows, 2, 4))

            class _Out:
                last_hidden_state = hidden

            return _Out()

    tokenizer = _Tokenizer()

    class _Manager:
        def get_embedding_model(self):
            return tokenizer, _Model(), torch.device("cpu"), 4

    monkeypatch.setattr("app.core.state.get_model_manager", lambda: _Manager())
    monkeypatch.setattr("app.api.v1.endpoints.embedding.get_model_manager", lambda: _Manager())
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    client = EmbeddingClient("local://medembed")
    text = "Call 555-234-5678. Creatinine is 1.4. Continue metformin."
    await client.embed([text])
    sent = tokenizer.batches[0][0]
    assert "555-234-5678" not in sent
    assert "[REDACTED_PHONE]" in sent
    assert "Creatinine is 1.4." in sent
    assert "metformin" in sent

    from app.api.v1.endpoints.embedding import _local_embed

    _local_embed([text], True)
    direct = tokenizer.batches[1][0]
    assert "555-234-5678" not in direct
    assert "Creatinine is 1.4." in direct


@pytest.mark.asyncio
async def test_ready_reports_local_reranker_model(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    body = await pipelines_ready()
    assert body["reranker_model"] == settings.active_reranker_model()
    assert "bge-reranker-v2-m3" in body["reranker_model"]
    assert body["cloud_mode"] is False


@pytest.mark.asyncio
async def test_specialists_share_one_prefix_and_keep_allowed_chunk_scope(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    monkeypatch.setenv("GENERATION_BACKEND", "bedrock")
    monkeypatch.delenv("AGENT_PASSAGE_LIMIT", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_MAX_CHARS", raising=False)
    passages = [
        _passage("emr-1", SourceType.EMR, "Patient takes metformin."),
        _passage("cpg-1", SourceType.CPG, "Guideline recommends metformin first-line."),
        _passage("lit-1", SourceType.LIT, "Trial supports metformin.", safety_score=0.4),
        _passage("safety-1", SourceType.SAFETY, "Contraindicated when eGFR is below 30."),
    ]
    facets = [ClinicalFacet(name="safety_contraindications"), ClinicalFacet(name="guideline_concordance")]
    agents = {
        "emr": EMRAgent(_RecordingLLM()),
        "guideline": GuidelineAgent(_RecordingLLM()),
        "safety": SafetyAgent(_RecordingLLM()),
        "quality": QualityAgent(_RecordingLLM()),
        "critic": ContradictionAgent(_RecordingLLM()),
    }
    batches = {
        "emr": await agents["emr"].explore("metformin", passages, facets, patient_id="patient-7"),
        "guideline": await agents["guideline"].explore("metformin", passages, facets),
        "safety": await agents["safety"].explore("metformin", passages, facets),
        "quality": await agents["quality"].explore("metformin", passages, facets),
        "critic": await agents["critic"].explore("metformin", passages, facets),
    }
    prefixes = [_shared_prefix(agent.client.calls[0]["messages"]) for agent in agents.values()]
    system, emr_head = prefixes[0]
    assert system == shared_claim_system_message()
    assert system.startswith("You are a MedSwin specialist")
    assert "Return valid JSON matching this schema" in system
    assert emr_head.startswith("Shared Evidence Passages:\n")
    assert "emr-1" in emr_head
    assert "cpg-1" not in emr_head
    assert "safety-1" not in emr_head
    guideline_head = prefixes[1][1]
    assert "cpg-1" in guideline_head and "lit-1" in guideline_head
    assert "emr-1" not in guideline_head
    safety_head = prefixes[2][1]
    assert "safety-1" in safety_head and "lit-1" in safety_head
    assert "emr-1" not in safety_head
    assert {claim.chunk_id for claim in batches["emr"].claims} == {"emr-1"}
    assert {claim.chunk_id for claim in batches["guideline"].claims} == {"cpg-1", "lit-1"}
    assert {claim.chunk_id for claim in batches["safety"].claims} == {"safety-1", "lit-1"}
    assert {claim.chunk_id for claim in batches["quality"].claims} == {"emr-1", "cpg-1", "lit-1", "safety-1"}
    emr_suffix = agents["emr"].client.calls[0]["messages"][1]["content"].split(SPECIALIST_SUFFIX_MARK, 1)[1]
    assert "Query: metformin" in emr_suffix
    assert "Patient ID: patient-7" in emr_suffix
    assert "Valid chunk_ids: emr-1" in emr_suffix
    assert "cpg-1" not in emr_suffix.split("Valid chunk_ids:", 1)[1]
    assert "source_type EMR" in emr_suffix


@pytest.mark.asyncio
async def test_emr_specialist_with_no_chart_does_not_read_the_other_passages():
    agent = EMRAgent(_RecordingLLM())
    passages = [_passage("cpg-1", SourceType.CPG, "Guideline recommends metformin.")]
    batch = await agent.explore("metformin", passages, [ClinicalFacet(name="patient_applicability")])
    assert batch.claims == []
    assert agent.client.calls == []
    assert "no passages" in " ".join(batch.notes).lower()


def test_a_late_caution_stays_in_the_model_passage():
    from app.agents.base import passage_context
    from app.agents.synthesis import _bounded_local_evidence

    filler = "History note. " * 200
    passage = _passage("emr-late", SourceType.EMR, f"{filler} Avoid apixaban.")
    rendered = passage_context([passage])
    evidence, shown = _bounded_local_evidence([passage])
    assert rendered.lower().startswith("[emr-late")
    assert "Avoid apixaban." in rendered
    assert "Avoid apixaban." in evidence
    assert shown == ["emr-late"]


def test_the_model_passage_drops_a_phone_number():
    from app.agents.base import passage_context
    from app.agents.synthesis import _bounded_local_evidence

    passage = _passage(
        "emr-1",
        SourceType.EMR,
        "Call 555-234-5678. Creatinine is 1.4. Continue metformin.",
    )
    rendered = passage_context([passage])
    evidence, shown = _bounded_local_evidence([passage])
    assert "555-234-5678" not in rendered
    assert "[REDACTED_PHONE]" in rendered
    assert "Creatinine is 1.4." in rendered
    assert "metformin" in rendered
    assert "555-234-5678" not in evidence
    assert "[REDACTED_PHONE]" in evidence
    assert shown == ["emr-1"]


def test_local_passage_context_keeps_safety_inside_2048_envelope(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setattr(settings, "GENERATION_BACKEND", None)
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_LIMIT", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_MAX_CHARS", raising=False)
    filler = "alpha " * 400
    passages = [
        _passage(f"lit-{index}", SourceType.LIT, filler, calibrated_score=0.99 - (index * 0.01))
        for index in range(8)
    ]
    passages.append(
        _passage(
            "safety-keep",
            SourceType.SAFETY,
            "Contraindicated. Do not use this drug. " + filler,
            calibrated_score=0.01,
            contradiction_score=0.2,
        )
    )
    rendered = passage_context(passages)
    assert len(rendered) <= 1400
    assert rendered.count("[") <= 4
    assert "[safety-keep|SAFETY|doc=doc-safety-keep]" in rendered
    assert rendered.index("safety-keep") < rendered.index("lit-0")

    monkeypatch.setenv("AGENT_PASSAGE_LIMIT", "2")
    explicit = passage_context(passages)
    assert "[safety-keep|SAFETY|doc=doc-safety-keep]" in explicit
    assert explicit.index("safety-keep") < explicit.index("lit-0")
    assert "[lit-1|LIT|doc=doc-lit-1]" not in explicit


def test_a_fifth_short_caution_stays_in_the_local_passage(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setattr(settings, "GENERATION_BACKEND", None)
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_LIMIT", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_MAX_CHARS", raising=False)
    drugs = ["apixaban", "warfarin", "heparin", "enoxaparin", "metformin"]
    passages = [
        _passage(f"lit-{index}", SourceType.LIT, "alpha " * 40, calibrated_score=0.99 - (index * 0.01))
        for index in range(4)
    ]
    passages.extend(
        _passage(f"caution-{drug}", SourceType.EMR, f"Avoid {drug}.", calibrated_score=0.1)
        for drug in drugs
    )
    rendered = passage_context(passages)
    lowered = rendered.lower()
    assert len(rendered) <= 1400
    for drug in drugs:
        assert f"avoid {drug}." in lowered


def test_a_later_drug_stays_when_earlier_cautions_are_long(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setattr(settings, "GENERATION_BACKEND", None)
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_LIMIT", raising=False)
    monkeypatch.delenv("AGENT_PASSAGE_MAX_CHARS", raising=False)
    body = "Avoid warfarin. " + ("monitoring plan details " * 80)
    passages = [
        _passage(f"c-{index}", SourceType.EMR, body, calibrated_score=0.2)
        for index in range(5)
    ]
    passages.append(_passage("apix", SourceType.EMR, "Avoid apixaban.", calibrated_score=0.1))
    rendered = passage_context(passages)
    lowered = rendered.lower()
    assert len(rendered) <= 1400
    assert "avoid warfarin." in lowered
    assert "avoid apixaban." in lowered


def test_a_seventh_short_caution_stays_in_the_local_answer():
    from app.agents.synthesis import _bounded_local_evidence

    drugs = ["apixaban", "warfarin", "heparin", "enoxaparin", "metformin", "aspirin", "dabigatran"]
    passages = [
        _passage(f"lit-{index}", SourceType.LIT, "alpha " * 40, calibrated_score=0.99 - (index * 0.01))
        for index in range(6)
    ]
    passages.extend(
        _passage(f"caution-{drug}", SourceType.EMR, f"Avoid {drug}.", calibrated_score=0.1)
        for drug in drugs
    )
    evidence, shown = _bounded_local_evidence(passages)
    lowered = evidence.lower()
    assert len(evidence) <= 2800
    for drug in drugs:
        assert f"avoid {drug}." in lowered
        assert f"caution-{drug}" in shown


def test_a_later_drug_stays_in_the_local_answer_when_earlier_cautions_are_long():
    from app.agents.synthesis import _bounded_local_evidence

    body = "Avoid warfarin. " + ("monitoring plan details " * 80)
    passages = [
        _passage(f"c-{index}", SourceType.EMR, body, calibrated_score=0.2)
        for index in range(7)
    ]
    passages.append(_passage("apix", SourceType.EMR, "Avoid apixaban.", calibrated_score=0.1))
    evidence, shown = _bounded_local_evidence(passages)
    lowered = evidence.lower()
    assert len(evidence) <= 2800
    assert "avoid warfarin." in lowered
    assert "avoid apixaban." in lowered
    assert "apix" in shown


@pytest.mark.asyncio
async def test_a_caution_id_stays_in_the_local_citation_list():
    caution = _passage("caution-apixaban", SourceType.EMR, "Avoid apixaban.", calibrated_score=0.1)
    literature = [
        _passage(f"lit-{index}-{'x' * 80}", SourceType.LIT, "A general note.", calibrated_score=0.99 - (index * 0.01))
        for index in range(6)
    ]

    class _Client:
        backend = "medswin_local"
        calls = []

        async def call_llm(self, messages, json_schema=None, **kwargs):
            self.calls.append(messages)
            return {"content": json.dumps({"answer": "See the note.", "evidence_used": [], "insufficient_evidence": False})}

    client = _Client()
    await SynthesisAgent(client).synthesize_with_artifact("apixaban", _bundle([*literature, caution]))
    user = client.calls[0][1]["content"]
    evidence = user.split("Evidence:\n", 1)[1].split("\n\nValid chunk_ids:", 1)[0]
    allowed = user.split("Valid chunk_ids:", 1)[1].split("\n", 1)[0]
    assert "Avoid apixaban." in evidence
    assert "caution-apixaban" in allowed


@pytest.mark.asyncio
async def test_a_later_long_caution_id_stays_on_the_citation_list():
    drugs = ["apixaban", "warfarin", "heparin", "enoxaparin", "metformin", "aspirin", "dabigatran"]
    cautions = [
        _passage(
            f"caution-{drug}-{'x' * 60}",
            SourceType.EMR,
            f"Avoid {drug}.",
            calibrated_score=0.1,
        )
        for drug in drugs
    ]

    class _Client:
        backend = "medswin_local"
        calls = []

        async def call_llm(self, messages, json_schema=None, **kwargs):
            self.calls.append(messages)
            return {"content": json.dumps({"answer": "See the note.", "evidence_used": [], "insufficient_evidence": False})}

    client = _Client()
    await SynthesisAgent(client).synthesize_with_artifact("bleeding", _bundle(cautions))
    user = client.calls[0][1]["content"]
    allowed = user.split("Valid chunk_ids:", 1)[1].split("\n", 1)[0]
    for passage in cautions:
        assert passage.chunk_id in allowed


@pytest.mark.asyncio
async def test_local_synthesis_bounds_evidence_and_reads_inline_chunk_ids():
    safety = _passage("safety-1", SourceType.SAFETY, "Contraindicated in anaphylaxis. " + ("warn " * 300), calibrated_score=0.2)
    cpg = _passage("cpg-1", SourceType.CPG, "First-line therapy. " + ("guide " * 300), calibrated_score=0.95)
    extra = [
        _passage(f"extra-{index}", SourceType.LIT, "noise " * 300, calibrated_score=0.5)
        for index in range(8)
    ]
    bundle = _bundle([cpg, safety, *extra])

    class _Client:
        backend = "medswin_local"
        calls = []

        async def call_llm(self, messages, json_schema=None, **kwargs):
            self.calls.append(messages)
            return {"content": json.dumps({"answer": "See the note.", "evidence_used": [], "insufficient_evidence": False})}

    client = _Client()
    agent = SynthesisAgent(client)
    await agent.synthesize_with_artifact("pneumonia", bundle, safety_notes=["Confirm the allergy record."])
    user = client.calls[0][1]["content"]
    evidence = user.split("Evidence:\n", 1)[1].split("\n\nValid chunk_ids:", 1)[0]
    assert len(evidence) <= 2800
    assert "safety-1" in evidence
    assert "Valid chunk_ids:" in user
    assert user.index("Safety notes:") > user.index("Evidence:")
    assert "Confirm the allergy record." in user
    assert user.index("Valid chunk_ids:") > user.index("Evidence:")

    rendered, provenance = SynthesisAgent(None)._render(
        {"answer": "Use [safety-1] before cpg-1. Never cite missing-9.", "evidence_used": []},
        bundle,
    )
    assert provenance.cited_chunk_ids == ["safety-1"]
    assert provenance.statements[0]["use"] == "inline cited supporting evidence"
    assert "missing-9" not in provenance.cited_chunk_ids
    assert "cpg-1" not in provenance.cited_chunk_ids

    hidden, hidden_provenance = SynthesisAgent(None)._render(
        {"answer": "Use [extra-0].", "evidence_used": ["extra-0"]},
        bundle,
        shown_ids={"safety-1", "cpg-1"},
    )
    assert hidden_provenance.cited_chunk_ids == []
    assert "extra-0" in hidden_provenance.rejected_chunk_ids
    assert "safety-1" in rendered

    explicit, explicit_provenance = SynthesisAgent(None)._render(
        {"answer": "Also see [cpg-1].", "evidence_used": ["safety-1", "fake"]},
        bundle,
    )
    assert explicit_provenance.cited_chunk_ids == ["safety-1"]
    assert explicit_provenance.rejected_chunk_ids == ["fake"]
    assert "cpg-1" not in explicit_provenance.cited_chunk_ids

    holder = []

    class _Cloud:
        backend = "bedrock"

        async def call_llm(self, messages, json_schema=None, **kwargs):
            holder.append(messages)
            return {"content": json.dumps({"answer": "Cloud answer.", "evidence_used": ["cpg-1"]})}

    await SynthesisAgent(_Cloud()).synthesize_with_artifact("pneumonia", _bundle([cpg]), safety_notes=[])
    cloud_user = holder[0][1]["content"]
    assert "Valid chunk_ids:" not in cloud_user
    assert "Safety notes: []" in cloud_user
    assert cpg.text in cloud_user


@pytest.mark.asyncio
async def test_the_cloud_prompt_puts_the_caution_before_the_filler():
    filler = _passage("fill", SourceType.CPG, "A general diet note.", calibrated_score=0.99)
    caution = _passage("caution", SourceType.SAFETY, "Avoid apixaban.", calibrated_score=0.60)
    caution.safety_score = 0.40
    seen = []

    class _Cloud:
        backend = "foundry"

        async def call_llm(self, messages, json_schema=None, **kwargs):
            seen.append(messages)
            return {"content": json.dumps({"answer": "See the caution.", "evidence_used": ["caution"]})}

    await SynthesisAgent(_Cloud()).synthesize_with_artifact("Which dose?", _bundle([filler, caution]))
    evidence = seen[0][1]["content"].split("Evidence:\n", 1)[1]
    assert evidence.index("[caution]") < evidence.index("[fill]")
    assert "Avoid apixaban." in evidence


@pytest.mark.asyncio
async def test_the_cloud_prompt_keeps_a_late_caution_inside_one_passage():
    filler = "History note. " * 80
    passage = _passage("emr-late", SourceType.EMR, f"{filler} Avoid apixaban.", calibrated_score=0.8)
    seen = []

    class _Cloud:
        backend = "foundry"

        async def call_llm(self, messages, json_schema=None, **kwargs):
            seen.append(messages)
            return {"content": json.dumps({"answer": "See the caution.", "evidence_used": ["emr-late"]})}

    await SynthesisAgent(_Cloud()).synthesize_with_artifact("Which dose?", _bundle([passage]))
    evidence = seen[0][1]["content"]
    assert evidence.index("Avoid apixaban.") < evidence.index("History note.")


@pytest.mark.asyncio
async def test_local_llm_defaults_and_clamps_completion_inside_2048(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setenv("GENERATION_BACKEND", "medswin_local")
    monkeypatch.delenv("LLM_DEFAULT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MEDSWIN_DEFAULT_MAX_TOKENS", raising=False)
    seen = {}

    async def _http(*args, **kwargs):
        seen["json"] = kwargs["json"]
        return _JsonResponse({"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    monkeypatch.setattr("app.services.adapters.llm.request_with_model_rate_limit", _http)
    client = LLMClient("http://127.0.0.1:8000/v1/chat/completions")
    assert client.backend == "medswin_local"
    await client.call_llm(
        [{"role": "user", "content": "Call 555-234-5678. Creatinine is 1.4. Continue metformin."}]
    )
    sent = seen["json"]["messages"][0]["content"]
    assert "555-234-5678" not in sent
    assert "[REDACTED_PHONE]" in sent
    assert "Creatinine is 1.4." in sent
    assert "metformin" in sent
    assert seen["json"]["max_tokens"] == 384

    await client.call_llm([{"role": "user", "content": "short question"}], max_tokens=32)
    assert seen["json"]["max_tokens"] == 32

    huge = "token " * 4000
    await client.call_llm([{"role": "user", "content": huge}])
    sent_tokens = _estimate_message_tokens(seen["json"]["messages"])
    assert sent_tokens < _estimate_message_tokens([{"role": "user", "content": huge}])
    assert sent_tokens + seen["json"]["max_tokens"] <= 1980
    assert seen["json"]["max_tokens"] >= 96

    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setenv("GENERATION_BACKEND", "foundry")
    monkeypatch.setenv("FOUNDRY_MODEL", "gpt-5.4")
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "foundry")
    foundry = LLMClient("https://example.test/v1/chat/completions", model="gpt-5.4", api_key="key")
    await foundry.call_llm([{"role": "user", "content": huge}], max_tokens=32)
    assert seen["json"]["max_completion_tokens"] == 32
    assert "max_tokens" not in seen["json"]


def test_the_chart_window_keeps_every_drug_that_fits():
    from app.services.adapters.llm import _chart_block

    drugs = [
        "apixaban",
        "warfarin",
        "heparin",
        "enoxaparin",
        "metformin",
        "aspirin",
        "dabigatran",
        "rivaroxaban",
    ]
    chart = " ".join(f"Avoid {drug}." for drug in drugs)
    content = f"Which dose?\n\nPatient chart:\n{chart}\n\nEvidence:\n[c1] A note."
    block = _chart_block(content, 80)
    lowered = block.lower()
    assert len(block.split("Patient chart:\n", 1)[1]) <= 80
    for drug in drugs:
        assert drug in lowered


def test_the_local_window_keeps_a_caution_after_a_long_passage_header():
    history = "history " * 1500
    chunk_id = "caution-apixaban-" + ("x" * 80)
    header = f"[{chunk_id}|EMR|doc=d1]"
    assert len(header) > 80
    content = (
        "Shared Evidence Passages:\n"
        f"{header}\nAvoid apixaban. {history}\n\n"
        "Specialist Instructions:\n"
        "You are the safety specialist.\n\n"
        "Query: Which dose?\n\n"
        f"Valid chunk_ids: {chunk_id}\n"
        "Cite only a chunk_id from Valid chunk_ids."
    )
    fitted = fit_local_messages(
        [
            {"role": "system", "content": "Return one JSON object. " * 40},
            {"role": "user", "content": content},
        ],
        384,
    )
    text = fitted[-1]["content"]
    assert "Avoid apixaban" in text
    assert chunk_id in text.split("Valid chunk_ids:", 1)[1]
    assert "history " * 40 not in text
    assert _estimate_message_tokens(fitted) + 384 <= 1980


def test_the_local_window_keeps_the_specialist_citation_list():
    history = "history " * 1500
    long_ids = ", ".join(f"lit-{index}-{'x' * 40}" for index in range(8))
    content = (
        "Shared Evidence Passages:\n"
        f"[caution-apixaban|EMR|doc=d1]\nAvoid apixaban. {history}\n\n"
        "Specialist Instructions:\n"
        "You are the safety specialist.\n\n"
        f"Query: Which dose? {history}\n\n"
        f"Valid chunk_ids: {long_ids}, caution-apixaban\n"
        "Cite only a chunk_id from Valid chunk_ids."
    )
    fitted = fit_local_messages(
        [
            {"role": "system", "content": "Return one JSON object. " * 40},
            {"role": "user", "content": content},
        ],
        384,
    )
    text = fitted[-1]["content"]
    allowed = text.split("Valid chunk_ids:", 1)[1]
    assert "Avoid apixaban" in text
    assert "You are the safety specialist." in text
    assert "caution-apixaban" in allowed
    assert "lit-0-" not in allowed
    assert "Cite only a chunk_id from Valid chunk_ids." in text
    assert "history " * 40 not in text
    assert _estimate_message_tokens(fitted) + 384 <= 1980


def test_the_local_window_keeps_the_chart_and_the_caution():
    history = "history " * 1500
    cautions = [
        "Avoid apixaban.",
        "Avoid warfarin.",
        "Avoid heparin.",
        "Avoid enoxaparin.",
        "Avoid metformin.",
    ]
    blocks = "\n\n".join(
        f"[c{index}|SAFETY|doc=d{index}]\n{line} {history}"
        for index, line in enumerate(cautions, start=1)
    )
    specialist = (
        "Shared Evidence Passages:\n"
        f"{blocks}\n\n"
        "Query: Which dose?\n\n"
        "Patient chart:\nASA 100mg daily\n"
        "The patient chart is context. Cite only a chunk_id from the evidence."
    )
    synthesis = (
        "Query: Which dose?\n\n"
        "Patient chart:\nASA 100mg daily\n"
        "The patient chart is context. Cite only a chunk_id from the evidence.\n\n"
        "Facet coverage:\n- guideline: satisfied\n\n"
        f"Evidence:\n[c1] Avoid apixaban. {history}\n\n"
        "Valid chunk_ids: c1\n"
    )
    for content in (specialist, synthesis):
        fitted = fit_local_messages(
            [
                {"role": "system", "content": "Return one JSON object. " * 40},
                {"role": "user", "content": content},
            ],
            384,
        )
        text = fitted[-1]["content"]
        assert "Avoid apixaban" in text
        if "ASA 100mg daily" in content:
            assert "ASA 100mg daily" in text
        if "Avoid metformin." in content:
            for line in cautions:
                assert line in text
        assert "history " * 40 not in text
        assert _estimate_message_tokens(fitted) + 384 <= 1980
    no_chart = (
        "Shared Evidence Passages:\n"
        f"{blocks}\n\n"
        "Query: Which dose?\n"
    )
    fitted = fit_local_messages(
        [
            {"role": "system", "content": "Return one JSON object. " * 40},
            {"role": "user", "content": no_chart},
        ],
        384,
    )
    text = fitted[-1]["content"]
    for line in cautions:
        assert line in text
    assert "history " * 40 not in text
    assert _estimate_message_tokens(fitted) + 384 <= 1980


def test_the_local_window_keeps_a_drug_past_the_first_eighty_characters():
    long_line = (
        "Avoid anticoagulation immediately whenever haematology confirms "
        "persistent gastrointestinal bleeding involving apixaban."
    )
    assert "apixaban" not in long_line[:80]
    history = "history " * 1500
    specialist = (
        "Shared Evidence Passages:\n"
        f"[c1|SAFETY|doc=d1]\n{long_line} {history}\n\n"
        "Query: Which dose?\n"
    )
    chart = (
        "Query: Which dose?\n\n"
        f"Patient chart:\n{long_line} {history}\n"
        "The patient chart is context.\n\n"
        "Evidence:\n"
        f"[c1] {long_line} {history}\n\n"
        "Valid chunk_ids: c1\n"
    )
    for content in (specialist, chart):
        fitted = fit_local_messages(
            [
                {"role": "system", "content": "Return one JSON object. " * 40},
                {"role": "user", "content": content},
            ],
            384,
        )
        text = fitted[-1]["content"].lower()
        assert "apixaban" in text
        assert "avoid" in text
        assert "history " * 40 not in text
        assert _estimate_message_tokens(fitted) + 384 <= 1980


def test_the_local_window_keeps_a_recommended_drug_past_the_first_eighty_characters():
    line = (
        "The guideline recommends continued oral anticoagulation for nonvalvular "
        "atrial fibrillation with apixaban."
    )
    assert "apixaban" not in line[:80]
    history = "history " * 1500
    prompts = [
        (
            "Shared Evidence Passages:\n"
            f"[c1|CPG|doc=d1]\n{history}\n{line}\n\n"
            "Query: Which anticoagulant?\n"
        ),
        (
            "Query: Which anticoagulant?\n\n"
            "Evidence:\n"
            f"[c1] {history} {line}\n\n"
            "Valid chunk_ids: c1\n"
        ),
    ]
    for content in prompts:
        fitted = fit_local_messages(
            [
                {"role": "system", "content": "Return one JSON object. " * 40},
                {"role": "user", "content": content},
            ],
            384,
        )
        text = fitted[-1]["content"].lower()
        assert "apixaban" in text
        assert "recommends" in text
        assert "history " * 40 not in text
        assert "aspirin" not in text
        assert _estimate_message_tokens(fitted) + 384 <= 1980


def test_nist_download_sends_a_browser_user_agent(monkeypatch, tmp_path):
    seen = {}

    class _Body(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def _urlopen(request, timeout=None):
        seen["agent"] = request.get_header("User-agent")
        seen["url"] = request.full_url
        return _Body(b"nist-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    dest = tmp_path / "topics2016.xml"
    _download_url("https://trec.nist.gov/data/clinical/topics2016.xml", dest)
    assert seen["agent"] == "Mozilla/5.0 (compatible; MedSwin-Eval/2.0)"
    assert seen["url"] == "https://trec.nist.gov/data/clinical/topics2016.xml"
    assert dest.read_bytes() == b"nist-bytes"


def test_generator_contract_resolves_bedrock_and_full_local_stack(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    cloud = resolve_generator_contract("cloud")
    assert cloud.backend == "bedrock"
    assert cloud.model == "global.openai.gpt-6-luna"
    assert cloud.embedding_model == "global.cohere.embed-v4:0"
    assert cloud.reranker_model == "cohere.rerank-v3-5:0"
    assert cloud.cloud_mode is True

    medswin = resolve_generator_contract("medswin")
    assert medswin.backend == "medswin_local"
    assert medswin.model == EXPECTED_LOCAL_MODEL
    assert medswin.embedding_model == "global.cohere.embed-v4:0"
    assert medswin.cloud_mode is True

    local = resolve_generator_contract("medswin", local_retrieval=True)
    assert local.cloud_mode is False
    assert local.backend == "medswin_local"
    assert local.model == "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
    assert local.embedding_model == "MedEmbed-large-v0.1"
    assert local.reranker_model == "bge-reranker-v2-m3"

    environment = _profile_environment(cloud, "trec")
    assert environment["CLOUD_PROVIDER"] == "bedrock"
    assert environment["GENERATION_BACKEND"] == "bedrock"
    assert environment["BEDROCK_CHAT_MODEL"] == "global.openai.gpt-6-luna"
    assert environment["BEDROCK_EMBED_MODEL"] == "global.cohere.embed-v4:0"
    assert environment["BEDROCK_RERANK_MODEL"] == "cohere.rerank-v3-5:0"
    assert _retrieval_identity_matches("loaded", local.embedding_model, local.cloud_mode) is True
    assert _retrieval_identity_matches("embed-v-4-0", cloud.embedding_model, True) is False

    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "foundry")
    assert resolve_generator_contract("cloud") == GENERATOR_CONTRACTS["cloud"]
    foundry_env = _profile_environment(GENERATOR_CONTRACTS["cloud"], "bioasq")
    assert foundry_env["CLOUD_PROVIDER"] == "foundry"
    assert foundry_env["FOUNDRY_MODEL"] == "gpt-5.4"
    assert foundry_env["CLOUD_EMBEDDING"] == "embed-v-4-0"
    assert foundry_env["CLOUD_RERANKER"] == "Cohere-rerank-v4.0-fast"
