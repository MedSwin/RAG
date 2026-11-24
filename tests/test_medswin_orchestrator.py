import numpy as np

from app.agents.base import parse_claims
from app.medswin.normalize import QueryNormalizer
from app.models.medswin import CandidatePassage, ClinicalFacet, ClinicalScope, QuerySpec, SourceType
from app.schemas.enums import PipelineId
from app.schemas.traces import RerankTrace, RetrievalTrace
from app.services.medswin.orchestrator import MedSwinOrchestrator


class FakeEmbeddingClient:
    async def embed(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32)]


class FakeRetriever:
    async def retrieve(self, **kwargs):
        passage = CandidatePassage(
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
        return [passage], RetrievalTrace(union_count=1)

    async def rerank(self, query, candidates):
        return candidates, RerankTrace(calibration_version="test")

    def fuse_and_select(self, candidates, facets=None, agent_weights=None):
        return candidates

    def rank(self, candidates, facets=None):
        return candidates

    def select(self, candidates, facets=None, agent_weights=None, token_budget=None):
        return candidates

    def build_bundle(self, passages, **kwargs):
        from app.models.medswin import EvidenceBundle

        return EvidenceBundle(
            passages=passages,
            total_tokens=20,
            cpg_count=1,
            emr_count=0,
            lit_count=0,
            safety_count=0,
            facet_coverage=kwargs.get("facet_coverage") or [],
            evidence_ledger=kwargs.get("evidence_ledger") or [],
            contradictions=kwargs.get("contradictions") or [],
            policy_decision=kwargs.get("policy_decision"),
        )


class EmptyRepo:
    async def get_by_patient_id(self, patient_id, org_id):
        return []


async def _noop_agents(*args, **kwargs):
    return []


def test_query_spec_coercion_recovers_malformed_llm_fields():
    data = QueryNormalizer(None).coerce_spec({
        "canonical_terms": ["heart failure"],
        "clinical_scope": "treatment evidence retrieval",
        "facets": [
            {
                "name": "treatment",
                "threshold": "relevant to intervention comparative effectiveness",
                "weight": "1.25",
                "keywords": "therapy",
            }
        ],
    })

    spec = QuerySpec(**data)

    assert spec.clinical_scope == ClinicalScope.CLINICIAN_CDS
    assert spec.facets[0].threshold == 0.70
    assert spec.facets[0].weight == 1.25
    assert spec.facets[0].keywords == ["therapy"]


def test_claim_parser_keeps_only_retrieved_chunks_and_canonical_facets():
    passage = CandidatePassage(
        chunk_id="c1",
        doc_id="d1",
        source_type=SourceType.CPG,
        text="Avoid therapy in severe renal impairment.",
        calibrated_score=0.8,
    )
    facets = [ClinicalFacet(name="safety_contraindications")]
    batch = parse_claims(
        "safety",
        {
            "claims": [
                {"facet": "contraindication", "claim": "Avoid in renal impairment", "chunk_id": "c1", "polarity": "safety"},
                {"facet": "safety", "claim": "hallucinated", "chunk_id": "missing", "polarity": "supports"},
            ]
        },
        [passage],
        facets=facets,
    )
    assert len(batch.claims) == 1
    assert batch.claims[0].facet == "safety_contraindications"
    assert batch.claims[0].chunk_id == "c1"


def test_retrieve_with_sufficiency_returns_policy_artifacts_for_insufficient_bundle():
    import asyncio

    orchestrator = MedSwinOrchestrator(embedding_client=FakeEmbeddingClient(), reranker_client=None)
    orchestrator.retriever = FakeRetriever()
    orchestrator.chunk_repo = EmptyRepo()
    orchestrator._dispatch_agents = _noop_agents
    from app.schemas.traces import AuditTrace

    trace = AuditTrace(trace_id="t1", session_id="s1", user_id="u1", org_id="org1", query="q", patient_id="patient1")
    bundle, final_check = asyncio.run(
        orchestrator._retrieve_with_sufficiency(
            query="What treatment is safe for this patient?",
            query_spec=QuerySpec(canonical_terms=["treatment"]),
            org_id="org1",
            patient_id="patient1",
            constraints={},
            trace=trace,
        )
    )

    assert bundle.policy_decision is not None
    assert bundle.policy_decision.passed is False
    assert bundle.facet_coverage
    assert trace.policy_decisions
    assert final_check is not None
    assert final_check.passed is False


def test_two_stage_skips_mac_but_keeps_gate():
    import asyncio

    dispatched = {"n": 0}

    orchestrator = MedSwinOrchestrator(
        embedding_client=FakeEmbeddingClient(), reranker_client=None, pipeline=PipelineId.TWO_STAGE
    )
    orchestrator.retriever = FakeRetriever()
    orchestrator.chunk_repo = EmptyRepo()
    from app.schemas.traces import AuditTrace

    trace = AuditTrace(trace_id="t4", session_id="s1", user_id="u1", org_id="org1", query="q", patient_id="patient1")
    bundle, final_check = asyncio.run(
        orchestrator._retrieve_with_sufficiency(
            query="What treatment is safe for this patient?",
            query_spec=QuerySpec(canonical_terms=["treatment"]),
            org_id="org1",
            patient_id="patient1",
            constraints={},
            trace=trace,
        )
    )

    assert dispatched["n"] == 0
    assert final_check is not None
    assert final_check.passed is False
    assert bundle.policy_decision is not None
    assert bundle.policy_decision.passed is False
    assert any(call.tool_name == "pipeline.mac" for call in trace.tool_calls)


def test_retrieval_packs_with_the_policy_token_budget(monkeypatch):
    import asyncio

    seen = {}

    def _pack(selected, facets=None, token_budget=None):
        seen["token_budget"] = token_budget
        return selected

    monkeypatch.setattr("app.medswin.orchestrator.pack_bundle", _pack)
    orchestrator = MedSwinOrchestrator(
        embedding_client=FakeEmbeddingClient(), reranker_client=None, pipeline=PipelineId.TWO_STAGE
    )
    orchestrator.policy = type(orchestrator.policy)(
        **{**orchestrator.policy.__dict__, "generation_token_budget": 222}
    )
    orchestrator.retriever = FakeRetriever()
    orchestrator.chunk_repo = EmptyRepo()
    from app.schemas.traces import AuditTrace

    trace = AuditTrace(trace_id="t5", session_id="s1", user_id="u1", org_id="org1", query="q")
    asyncio.run(
        orchestrator._retrieve_with_sufficiency(
            query="What treatment is safe?",
            query_spec=QuerySpec(canonical_terms=["treatment"]),
            org_id="org1",
            patient_id=None,
            constraints={},
            trace=trace,
        )
    )
    assert seen["token_budget"] == 222


def test_trace_failure_keeps_the_gated_answer():
    import asyncio

    from app.schemas.enums import PolicyAction
    from app.schemas.evidence import EvidenceBundle, PolicyDecision

    class BoomTrace:
        async def create(self, trace, org_id):
            raise RuntimeError("mongo down")

    class Sessions:
        async def create(self, session, org_id):
            return {}

    class Normalizer:
        async def normalize(self, query, trace):
            return QuerySpec(canonical_terms=["treatment"])

        def build_facets(self, query, query_spec, constraints, patient_id):
            return []

    orchestrator = MedSwinOrchestrator(
        embedding_client=FakeEmbeddingClient(), reranker_client=None, pipeline=PipelineId.TWO_STAGE
    )
    orchestrator.session_repo = Sessions()
    orchestrator.trace_repo = BoomTrace()
    orchestrator.normalizer = Normalizer()
    decision = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="No evidence.",
    )
    bundle = EvidenceBundle(
        passages=[],
        total_tokens=0,
        cpg_count=0,
        emr_count=0,
        lit_count=0,
        policy_decision=decision,
    )

    async def _retrieve(**kwargs):
        return bundle, type("Check", (), {"iteration": 0})()

    orchestrator._retrieve_with_sufficiency = _retrieve
    response = asyncio.run(orchestrator.chat(query="q", user_id="u1", org_id="org1"))
    assert response.degraded_mode.get("trace") is True
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert "insufficient" in response.answer.lower()


def test_session_failure_keeps_the_gated_answer():
    import asyncio

    from app.schemas.enums import PolicyAction
    from app.schemas.evidence import EvidenceBundle, PolicyDecision

    class BoomSession:
        async def create(self, session, org_id):
            raise RuntimeError("mongo down")

    class Traces:
        async def create(self, trace, org_id):
            return {}

    class Normalizer:
        async def normalize(self, query, trace):
            return QuerySpec(canonical_terms=["treatment"])

        def build_facets(self, query, query_spec, constraints, patient_id):
            return []

    orchestrator = MedSwinOrchestrator(
        embedding_client=FakeEmbeddingClient(), reranker_client=None, pipeline=PipelineId.TWO_STAGE
    )
    orchestrator.session_repo = BoomSession()
    orchestrator.trace_repo = Traces()
    orchestrator.normalizer = Normalizer()
    decision = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="No evidence.",
    )
    bundle = EvidenceBundle(
        passages=[],
        total_tokens=0,
        cpg_count=0,
        emr_count=0,
        lit_count=0,
        policy_decision=decision,
    )

    async def _retrieve(**kwargs):
        return bundle, type("Check", (), {"iteration": 0})()

    orchestrator._retrieve_with_sufficiency = _retrieve
    response = asyncio.run(orchestrator.chat(query="q", user_id="u1", org_id="org1"))
    assert response.degraded_mode.get("session") is True
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert "insufficient" in response.answer.lower()

