import asyncio

import numpy as np

from app.medswin.naive import NaiveRAGOrchestrator, compare_responses
from app.schemas.enums import PolicyAction, SourceType
from app.schemas.evidence import CandidatePassage, EvidenceBundle, PolicyDecision
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

    assert response.pipeline == "naive_rag"
    assert response.retrieval_backend == "ann"
    assert len(response.evidence_bundle.passages) == 1
    assert response.evidence_bundle.passages[0].chunk_id == "c1"
    assert response.policy_decision is not None
    assert response.policy_decision.passed is True
    assert response.policy_decision.action == PolicyAction.ACCEPT
    assert "sufficiency" in (response.safety_notes or "").lower()
    assert response.answer == "Naive answer from top-K passages."
    assert llm.calls
    assert dense.calls[0]["k"] == 1
    assert response.retrieval_traces[0].lexical_count == 0
    assert response.rerank_traces == []


def test_naive_does_not_call_a_reranker_client():
    orch, _, _ = _orchestrator()
    assert getattr(orch, "reranker_client", None) is None
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.rerank_traces == []
    assert all(passage.rerank_score is None for passage in response.evidence_bundle.passages)


def test_naive_generates_when_retrieval_is_empty():
    orch, _, llm = _orchestrator(passages=[])

    async def _empty(*args, **kwargs):
        return [], 0

    orch._mongo_cosine = _empty
    response = asyncio.run(orch.chat(query="q", user_id="u1", org_id="org1"))
    assert response.pipeline == "naive_rag"
    assert response.evidence_bundle.passages == []
    assert response.policy_decision.passed is True
    assert "No passages were retrieved" in llm.calls[0]["messages"][0]["content"]


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
    assert "embeddings" in response.answer.lower()
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
        pipeline="naive_rag",
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
        pipeline="medswin",
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
    assert diff["jaccard"] == 1 / 3
