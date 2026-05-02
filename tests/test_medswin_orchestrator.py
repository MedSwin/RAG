import numpy as np
import pytest

from app.models.medswin import AuditTrace, CandidatePassage, EvidenceBundle, QuerySpec, SourceType
from app.services.medswin.orchestrator import MedSwinOrchestrator


class FakeEmbeddingClient:
    async def embed(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32)]


class FakeRetrievalPipeline:
    async def retrieve(self, **kwargs):
        return [
            CandidatePassage(
                chunk_id="c1",
                doc_id="d1",
                source_type=SourceType.CPG,
                text="Guideline recommendation supports therapy.",
                section="Recommendations",
                calibrated_score=0.9,
                rerank_score=0.9,
                fusion_score=0.9,
                evidence_grade_score=0.95,
            )
        ]

    async def rerank(self, query, candidates):
        return candidates

    def compute_fusion_scores(self, candidates):
        return candidates

    def select_with_mmr(self, candidates, query_embedding, facets=None):
        return candidates

    def build_evidence_bundle(self, passages, **kwargs):
        return EvidenceBundle(
            passages=passages,
            total_tokens=20,
            cpg_count=1,
            emr_count=0,
            lit_count=0,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_retrieve_with_sufficiency_returns_policy_artifacts_for_insufficient_bundle():
    orchestrator = MedSwinOrchestrator(embedding_client=FakeEmbeddingClient(), reranker_client=None)
    orchestrator.embedding_client = FakeEmbeddingClient()
    orchestrator.retrieval_pipeline = FakeRetrievalPipeline()
    trace = AuditTrace(trace_id="t1", session_id="s1", user_id="u1", org_id="org1", query="q", patient_id="patient1")

    bundle = await orchestrator._retrieve_with_sufficiency(
        query="What treatment is safe for this patient?",
        query_spec=QuerySpec(canonical_terms=["treatment"]),
        org_id="org1",
        patient_id="patient1",
        constraints={},
        trace=trace,
    )

    assert bundle.policy_decision is not None
    assert bundle.policy_decision.passed is False
    assert bundle.facet_coverage
    assert trace.policy_decisions
