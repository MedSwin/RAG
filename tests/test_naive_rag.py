import asyncio

import numpy as np

from app.medswin.naive import NAIVE_POLICY_REASON, NaiveRAGOrchestrator, compare_responses
from app.schemas.enums import PipelineId, PolicyAction, SourceType
from app.schemas.evidence import AnswerArtifact, CandidatePassage, EvidenceBundle, PolicyDecision
from app.schemas.traces import ChatResponse


class FakeEmbeddingClient:
    async def embed(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32)]


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def call_llm(self, messages, temperature=0.2, **kwargs):
        self.calls.append({"messages": messages, "temperature": temperature})
        return {"content": "Naive answer from top-K passages."}


class FakeDense:
    def __init__(self, passages=None):
        self.calls = []
        self.passages = passages if passages is not None else [
            CandidatePassage(
                chunk_id="c1",
                doc_id="d1",
                source_type=SourceType.LIT,
                text="Metformin is usually continued when eGFR is stable.",
                dense_score=0.91,
            ),
            CandidatePassage(
                chunk_id="c2",
                doc_id="d2",
                source_type=SourceType.CPG,
                text="Hold metformin below the labelled eGFR threshold.",
                dense_score=0.44,
            ),
        ]

    async def retrieve(self, query_embedding, org_id, k, source_type_filter, patient_id, constraints):
        self.calls.append(
            {
                "org_id": org_id,
                "k": k,
                "source_type_filter": source_type_filter,
                "patient_id": patient_id,
                "constraints": constraints,
            }
        )
        return list(self.passages)


class EmptyRepo:
    async def create(self, *args, **kwargs):
        return {}

    async def update_last_active(self, *args, **kwargs):
        return True


class RecordingTraceRepo:
    def __init__(self):
        self.traces = []

    async def create(self, trace, org_id):
        self.traces.append((trace, org_id))
        return {}


def _orchestrator(passages=None):
    dense = FakeDense(passages)
    llm = FakeLLM()
    orch = NaiveRAGOrchestrator(
        embedding_client=FakeEmbeddingClient(),
        llm_client=llm,
        dense_retriever=dense,
    )
    orch.session_repo = EmptyRepo()
    orch.trace_repo = RecordingTraceRepo()
    return orch, dense, llm


def test_naive_retrieves_dense_top_k_and_always_generates():
    orch, dense, llm = _orchestrator()
    response = asyncio.run(
        orch.chat(
            query="Can metformin continue?",
            user_id="u1",
            org_id="org1",
            patient_id="p1",
            top_k=1,
        )
    )

    assert response.pipeline == PipelineId.NAIVE
    assert response.retrieval_backend == "ann"
    assert len(response.evidence_bundle.passages) == 1
    assert response.evidence_bundle.passages[0].chunk_id == "c1"
    assert response.policy_decision is not None
    assert response.policy_decision.passed is True
    assert response.policy_decision.action == PolicyAction.ACCEPT
    assert "sufficiency" in (response.safety_notes or "").lower()
    assert response.answer.startswith("Clinician decision support only:")
    assert response.answer.endswith("Naive answer from top-K passages.")
    assert llm.calls
    assert dense.calls[0]["k"] == 20
    assert len(response.ranked_evidence) == 2
    assert response.retrieval_traces[0].lexical_count == 0
    assert response.rerank_traces == []


def test_naive_does_not_call_a_reranker_client():
    orch, _, _ = _orchestrator()
    assert getattr(orch, "reranker_client", None) is None
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.rerank_traces == []
    assert all(passage.rerank_score is None for passage in response.evidence_bundle.passages)


def test_naive_abstains_when_retrieval_is_empty():
    orch, _, llm = _orchestrator(passages=[])

    async def _empty(*args, **kwargs):
        return [], 0

    orch._mongo_cosine = _empty
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.pipeline == PipelineId.NAIVE
    assert response.evidence_bundle.passages == []
    assert response.policy_decision.passed is False
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert response.answer_artifact.abstained is True
    assert "insufficient" in response.answer.lower()
    assert llm.calls == []


def test_naive_keeps_the_answer_when_the_session_write_fails():
    orch, _, llm = _orchestrator(passages=[])

    class BoomSession:
        async def create(self, session, org_id):
            raise RuntimeError("mongo down")

    async def _empty(*args, **kwargs):
        return [], 0

    orch.session_repo = BoomSession()
    orch._mongo_cosine = _empty
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.degraded_mode.get("session") is True
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert "insufficient" in response.answer.lower()
    assert llm.calls == []


def test_naive_refuses_to_pretend_rag_when_corpus_has_no_embeddings():
    orch, _, llm = _orchestrator(passages=[])

    async def _empty(*args, **kwargs):
        return [], 0

    async def _diagnosis(*args, **kwargs):
        return {"chunk_count": 12, "embedded_count": 0, "index_exists": False}

    orch._mongo_cosine = _empty
    orch._diagnose_corpus = _diagnosis
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.degraded_mode.get("no_embeddings") is True
    assert response.policy_decision.passed is False
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert response.answer_artifact.abstained is True
    assert "embeddings" in response.answer.lower()
    assert llm.calls == []


def test_naive_error_does_not_return_accept():
    orch, _, llm = _orchestrator(passages=[])

    async def _boom(*args, **kwargs):
        raise RuntimeError("embedder timeout 555-234-5678")

    orch.embedding_client.embed = _boom
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.degraded_mode.get("error") is True
    assert response.policy_decision.passed is False
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert response.policy_decision.reason == "Naive RAG failed: RuntimeError."
    assert response.safety_notes == "RuntimeError"
    assert response.answer_artifact.abstained is True
    assert "555-234-5678" not in response.answer
    assert "555-234-5678" not in (response.safety_notes or "")
    assert "555-234-5678" not in response.policy_decision.reason
    assert llm.calls == []


def test_mongo_cosine_skips_dimension_mismatch():
    query = np.array([1.0, 0.0], dtype=np.float32)
    from app.medswin.naive import _cosine

    assert _cosine(query, [1.0, 0.0]) > 0.9
    assert _cosine(query, [0.1, 0.2, 0.3]) == 0.0


def test_compare_responses_reports_overlap_and_medswin_abstention():
    naive = ChatResponse(
        answer="naive",
        evidence_bundle=EvidenceBundle(
            passages=[
                CandidatePassage(chunk_id="a", doc_id="d1", source_type=SourceType.LIT, text="a"),
                CandidatePassage(chunk_id="b", doc_id="d2", source_type=SourceType.LIT, text="b"),
            ],
            total_tokens=2,
            cpg_count=0,
            emr_count=0,
            lit_count=2,
        ),
        trace_id="n1",
        pipeline=PipelineId.NAIVE,
        retrieval_backend="ann",
        policy_decision=PolicyDecision(passed=True, action=PolicyAction.ACCEPT, reason="naive"),
    )
    medswin = ChatResponse(
        answer="insufficient",
        evidence_bundle=EvidenceBundle(
            passages=[
                CandidatePassage(chunk_id="b", doc_id="d2", source_type=SourceType.LIT, text="b"),
                CandidatePassage(chunk_id="c", doc_id="d3", source_type=SourceType.SAFETY, text="c"),
            ],
            total_tokens=2,
            cpg_count=0,
            emr_count=0,
            lit_count=1,
        ),
        trace_id="m1",
        pipeline=PipelineId.FULL,
        policy_decision=PolicyDecision(
            passed=False,
            action=PolicyAction.INSUFFICIENT_EVIDENCE,
            reason="missing critical facets",
        ),
    )
    diff = compare_responses(naive, medswin, naive_ms=10.0, medswin_ms=40.0)
    assert diff["overlap_chunk_ids"] == ["b"]
    assert diff["naive_only_chunk_ids"] == ["a"]
    assert diff["medswin_only_chunk_ids"] == ["c"]
    assert diff["medswin_abstained"] is True
    crashed = medswin.model_copy(update={"policy_decision": None, "answer_artifact": AnswerArtifact(content="", rendered_text="", abstained=True)})
    assert compare_responses(naive, crashed)["medswin_abstained"] is True
    missing = medswin.model_copy(update={"policy_decision": None, "answer_artifact": None})
    missing_diff = compare_responses(naive, missing)
    assert missing_diff["medswin_abstained"] is True
    assert missing_diff["medswin_policy_action"] is None
    accepted = medswin.model_copy(update={
        "policy_decision": PolicyDecision(passed=True, action=PolicyAction.ACCEPT, reason="accepted"),
        "answer_artifact": None,
    })
    assert compare_responses(naive, accepted)["medswin_abstained"] is False
    naive_stop = naive.model_copy(update={"answer_artifact": AnswerArtifact(content="", rendered_text="", abstained=True)})
    assert compare_responses(naive_stop, medswin)["naive_always_generated"] is False
    assert diff["jaccard"] == 1 / 3
    assert "empty context" not in NAIVE_POLICY_REASON
    assert "only when dense retrieval returns passages" in NAIVE_POLICY_REASON


def test_naive_search_and_answer_drop_the_phone():
    class RecordingEmbed:
        def __init__(self):
            self.texts = []

        async def embed(self, texts):
            self.texts.extend(texts)
            return [np.array([1.0, 0.0], dtype=np.float32)]

    embed = RecordingEmbed()
    dense = FakeDense()
    llm = FakeLLM()
    orch = NaiveRAGOrchestrator(
        embedding_client=embed,
        llm_client=llm,
        dense_retriever=dense,
    )
    orch.session_repo = EmptyRepo()
    orch.trace_repo = RecordingTraceRepo()
    asyncio.run(
        orch.chat(query="Call 555-234-5678 about metformin", user_id="u1", org_id="org1")
    )
    assert embed.texts
    assert "555-234-5678" not in embed.texts[0]
    assert "[REDACTED_PHONE]" in embed.texts[0]
    assert "metformin" in embed.texts[0]
    blob = str(llm.calls)
    assert "555-234-5678" not in blob
    assert "[REDACTED_PHONE]" in blob


def test_a_late_caution_stays_in_the_naive_context():
    from app.medswin.naive import _truncate_context

    fillers = [
        CandidatePassage(
            chunk_id=f"fill-{index}",
            doc_id=f"fill-{index}",
            source_type=SourceType.CPG,
            text="diet " * 400,
            calibrated_score=0.99,
        )
        for index in range(3)
    ]
    caution = CandidatePassage(
        chunk_id="caution",
        doc_id="caution",
        source_type=SourceType.SAFETY,
        text="Avoid apixaban.",
        calibrated_score=0.60,
        safety_score=0.40,
    )
    packed = _truncate_context(fillers + [caution], max_chars=500, token_budget=80)
    assert packed[0].chunk_id == "caution"
    assert "Avoid apixaban." in packed[0].text


def test_the_naive_claim_keeps_a_late_caution():
    filler = "History note. " * 40
    passage = CandidatePassage(
        chunk_id="emr-late",
        doc_id="emr-late",
        source_type=SourceType.EMR,
        text=f"{filler}Avoid apixaban.",
        dense_score=0.8,
    )
    ledger = NaiveRAGOrchestrator._naive_ledger(object(), [passage])
    claim = ledger[0].claims[0].claim
    assert claim.startswith("Avoid apixaban.")
    assert "History note." in claim


def test_the_naive_prompt_keeps_a_late_caution():
    class RecordingEmbed:
        async def embed(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32)]

    filler = "History note. " * 300
    passage = CandidatePassage(
        chunk_id="emr-late",
        doc_id="emr-late",
        source_type=SourceType.EMR,
        text=f"{filler} Avoid apixaban.",
        dense_score=0.9,
    )
    llm = FakeLLM()
    llm.backend = "medswin_local"
    orch = NaiveRAGOrchestrator(
        embedding_client=RecordingEmbed(),
        llm_client=llm,
        dense_retriever=FakeDense([passage]),
    )
    orch.session_repo = EmptyRepo()
    orch.trace_repo = RecordingTraceRepo()
    asyncio.run(orch.chat(query="Which dose?", user_id="u1", org_id="org1"))
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Avoid apixaban." in prompt
    assert prompt.index("Avoid apixaban.") < prompt.index("History note.")


def test_naive_passage_text_drops_the_phone():
    class RecordingEmbed:
        async def embed(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32)]

    passage = CandidatePassage(
        chunk_id="emr-1",
        doc_id="emr-1",
        source_type=SourceType.EMR,
        text="Call 555-234-5678. Creatinine is 1.4. Continue metformin.",
        dense_score=0.9,
    )
    llm = FakeLLM()
    orch = NaiveRAGOrchestrator(
        embedding_client=RecordingEmbed(),
        llm_client=llm,
        dense_retriever=FakeDense([passage]),
    )
    orch.session_repo = EmptyRepo()
    orch.trace_repo = RecordingTraceRepo()
    asyncio.run(orch.chat(query="What is the creatinine?", user_id="u1", org_id="org1"))
    blob = str(llm.calls)
    assert "555-234-5678" not in blob
    assert "[REDACTED_PHONE]" in blob
    assert "Creatinine is 1.4." in blob
    assert "metformin" in blob
