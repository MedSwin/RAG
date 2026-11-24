"""End-to-end runtime proof of the MedSwin CDSS app and multi-agent control plane.

Uses deterministic in-memory fakes for MongoDB, embeddings, reranking, and LLM
calls so the full FastAPI application, Auth/RBAC middleware, three pipelines,
specialist MAC loop, sufficiency gate, retrieve-more iteration, contradiction
adjudication, synthesis grounding, trace/session audit, and CDS Hooks cards +
feedback are all exercised with zero Bedrock spend.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import AuthMiddleware
from app.api.cds import get_card_lookup, get_feedback_recorder, router as cds_router
from app.api.v1.endpoints import embedding as embedding_endpoint
from app.api.v1.endpoints import pipelines
from app.core.config import settings
from app.medswin.naive import NaiveRAGOrchestrator
from app.medswin.orchestrator import MedSwinOrchestrator
from app.schemas.enums import PipelineId, PolicyAction, SourceType
from app.schemas.evidence import CandidatePassage, QuerySpec
from app.schemas.facets import ClinicalFacet
from app.schemas.traces import RerankTrace, RetrievalTrace


def _sign(secret: str, payload: Dict[str, Any]) -> str:
    def _enc(obj: Dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    head = _enc({"alg": "HS256", "typ": "JWT"})
    body = _enc(payload)
    sig = hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


class _MemorySessions:
    def __init__(self) -> None:
        self.rows: Dict[tuple[str, str], Dict[str, Any]] = {}

    async def create(self, session, org_id: str) -> Dict[str, Any]:
        doc = session.model_dump()
        self.rows[(session.session_id, org_id)] = doc
        return doc

    async def update_last_active(self, session_id: str, org_id: str, user_id: str | None = None) -> None:
        if (session_id, org_id) in self.rows:
            self.rows[(session_id, org_id)]["last_active"] = "updated"

    async def get_by_id(self, session_id: str, org_id: str):
        return self.rows.get((session_id, org_id))


class _MemoryTraces:
    def __init__(self) -> None:
        self.rows: Dict[tuple[str, str], Dict[str, Any]] = {}

    async def create(self, trace, org_id: str) -> Dict[str, Any]:
        doc = trace.model_dump()
        self.rows[(trace.trace_id, org_id)] = doc
        return doc

    async def get_by_id(self, trace_id: str, org_id: str):
        return self.rows.get((trace_id, org_id))


class _FakeEmbedder:
    async def embed(self, texts, input_type=None):
        return [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in texts]

    async def close(self):
        return None


class _FakeNormalizer:
    async def normalize(self, query: str, trace) -> QuerySpec:
        return QuerySpec(canonical_terms=["antibiotic", "pneumonia", "allergy"])

    def build_facets(self, query, query_spec, constraints, patient_id):
        return [
            ClinicalFacet(
                name="guideline_concordance",
                required=True,
                threshold=0.55,
                keywords=["amoxicillin", "guideline", "first-line", "pneumonia"],
            ),
            ClinicalFacet(
                name="safety_contraindications",
                required=True,
                threshold=0.55,
                keywords=["allergy", "anaphylaxis", "contraindicated", "avoid"],
            ),
        ]


class _StagedRetriever:
    """Iteration 0 returns only CPG (missing safety); iteration 1 adds SAFETY."""

    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.calls: List[Dict[str, Any]] = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.empty:
            return [], RetrievalTrace(union_count=0)
        hints = kwargs.get("hints") or {}
        cpg = CandidatePassage(
            chunk_id="cpg-1",
            doc_id="doc-cpg-1",
            source_type=SourceType.CPG,
            text="First-line guideline recommends amoxicillin for uncomplicated community pneumonia.",
            section="Recommendations",
            dense_score=0.88,
            lexical_score=0.80,
            rerank_score=0.90,
            calibrated_score=0.90,
            fusion_score=0.90,
            evidence_grade_score=0.92,
            token_count=18,
        )
        if not hints:
            return [cpg], RetrievalTrace(dense_count=1, lexical_count=1, union_count=1)
        safety = CandidatePassage(
            chunk_id="safety-1",
            doc_id="doc-safety-1",
            source_type=SourceType.SAFETY,
            text="Contraindicated in severe penicillin allergy or prior anaphylaxis; avoid beta-lactams.",
            section="Contraindications",
            dense_score=0.85,
            lexical_score=0.84,
            rerank_score=0.88,
            calibrated_score=0.88,
            fusion_score=0.88,
            evidence_grade_score=0.85,
            token_count=16,
        )
        return [cpg, safety], RetrievalTrace(dense_count=2, lexical_count=2, union_count=2)

    async def rerank(self, query, candidates):
        return candidates, RerankTrace(calibration_version="test-e2e")

    def rank(self, candidates, facets=None):
        return candidates

    def select(self, candidates, facets=None, agent_weights=None, token_budget=None):
        return candidates

    def build_bundle(self, passages, **kwargs):
        from app.schemas.evidence import EvidenceBundle

        return EvidenceBundle(
            passages=passages,
            total_tokens=sum(p.token_count or 10 for p in passages),
            cpg_count=sum(1 for p in passages if p.source_type == SourceType.CPG),
            emr_count=sum(1 for p in passages if p.source_type == SourceType.EMR),
            lit_count=sum(1 for p in passages if p.source_type == SourceType.LIT),
            safety_count=sum(1 for p in passages if p.source_type == SourceType.SAFETY),
            facet_coverage=kwargs.get("facet_coverage") or [],
            evidence_ledger=kwargs.get("evidence_ledger") or [],
            contradictions=kwargs.get("contradictions") or [],
            policy_decision=kwargs.get("policy_decision"),
        )


class _RoleLLM:
    def __init__(self, role: str) -> None:
        self.role = role
        self.backend = "bedrock"
        self.calls: List[List[Dict[str, str]]] = []

    async def call_llm(self, messages, json_schema=None, temperature=0.7, max_tokens=None, request_id=None):
        self.calls.append(messages)
        user_text = messages[-1]["content"]
        if self.role == "supervisor":
            return {
                "content": json.dumps(
                    {
                        "answer": "Amoxicillin is first-line unless severe penicillin allergy is present.",
                        "evidence_used": [
                            {"chunk_id": "cpg-1", "use": "first-line therapy"},
                            {"chunk_id": "safety-1", "use": "allergy contraindication"},
                            {"chunk_id": "fabricated-999", "use": "should be rejected"},
                        ],
                        "uncertainty": "Moderate; confirm allergy history before prescribing.",
                        "contraindications_risks": ["Avoid in severe penicillin anaphylaxis."],
                        "next_steps": ["Verify patient allergy record."],
                        "insufficient_evidence": False,
                    }
                )
            }
        if self.role == "safety" and "safety-1" in user_text:
            return {
                "content": json.dumps(
                    {
                        "claims": [
                            {
                                "facet": "safety_contraindications",
                                "claim": "Avoid beta-lactams in severe penicillin allergy.",
                                "polarity": "safety",
                                "chunk_id": "safety-1",
                                "confidence": 0.91,
                            }
                        ],
                        "notes": ["Allergy check verified."],
                    }
                )
            }
        if self.role == "guideline" and "cpg-1" in user_text:
            return {
                "content": json.dumps(
                    {
                        "claims": [
                            {
                                "facet": "guideline_concordance",
                                "claim": "Amoxicillin is recommended first-line.",
                                "polarity": "supports",
                                "chunk_id": "cpg-1",
                                "confidence": 0.92,
                            }
                        ],
                        "notes": [],
                    }
                )
            }
        if self.role == "quality" and "cpg-1" in user_text:
            return {
                "content": json.dumps(
                    {
                        "claims": [
                            {
                                "facet": "guideline_concordance",
                                "claim": "High-grade guideline passage.",
                                "polarity": "supports",
                                "chunk_id": "cpg-1",
                                "confidence": 0.88,
                            }
                        ],
                        "notes": [],
                    }
                )
            }
        return {"content": json.dumps({"claims": [], "notes": []})}

    async def close(self):
        return None


class _FakeNaiveDense:
    def __init__(self, passages: List[CandidatePassage]) -> None:
        self.passages = passages

    async def retrieve(self, *args, **kwargs):
        return list(self.passages)


class _EmptyChunkRepo:
    async def get_by_patient_id(self, patient_id: str, org_id: str):
        return []


def _wire_orchestrator(
    pipeline: PipelineId,
    sessions: _MemorySessions,
    traces: _MemoryTraces,
    *,
    empty: bool = False,
) -> tuple[MedSwinOrchestrator, Dict[str, _RoleLLM], _StagedRetriever]:
    orchestrator = MedSwinOrchestrator(
        embedding_client=_FakeEmbedder(),
        reranker_client=None,
        pipeline=pipeline,
    )
    llms: Dict[str, _RoleLLM] = {"supervisor": _RoleLLM("supervisor")}
    orchestrator.supervisor = llms["supervisor"]
    orchestrator.synthesis_agent.client = llms["supervisor"]
    orchestrator.normalizer = _FakeNormalizer()
    retriever = _StagedRetriever(empty=empty)
    orchestrator.retriever = retriever
    orchestrator.chunk_repo = _EmptyChunkRepo()
    orchestrator.session_repo = sessions
    orchestrator.trace_repo = traces
    if pipeline == PipelineId.FULL:
        for role, agent_attr in (
            ("emr", "emr_agent"),
            ("guideline", "guideline_agent"),
            ("safety", "safety_agent"),
            ("quality", "quality_agent"),
            ("critic", "critic_agent"),
        ):
            llms[role] = _RoleLLM(role)
            setattr(orchestrator, f"{role}_llm", llms[role])
            getattr(orchestrator, agent_attr).client = llms[role]
    return orchestrator, llms, retriever


def test_full_cdss_workflow_and_agentic_control_plane_end_to_end(monkeypatch):
    secret = "e2e-clinical-secret"
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setattr(settings, "CLOUD_PROVIDER", "bedrock")
    monkeypatch.setattr(settings, "AWS_BEDROCK_REGION", "ap-southeast-2")
    monkeypatch.setattr(settings, "BEDROCK_CHAT_MODEL", "global.openai.gpt-6-luna")
    monkeypatch.setattr(settings, "BEDROCK_EMBED_MODEL", "global.cohere.embed-v4:0")
    monkeypatch.setattr(settings, "BEDROCK_EMBED_DIMENSION", 1536)
    monkeypatch.setattr(settings, "BEDROCK_RERANK_MODEL", "cohere.rerank-v3-5:0")
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "RBAC_ALLOWED_ROLES", "clinician")

    sessions = _MemorySessions()
    traces = _MemoryTraces()
    full_orch, full_llms, full_retriever = _wire_orchestrator(PipelineId.FULL, sessions, traces)
    two_stage_orch, _, _ = _wire_orchestrator(PipelineId.TWO_STAGE, sessions, traces)

    naive_llm = _RoleLLM("naive")
    naive_orch = NaiveRAGOrchestrator(
        embedding_client=_FakeEmbedder(),
        llm_client=naive_llm,
        dense_retriever=_FakeNaiveDense([]),
    )
    naive_orch.session_repo = sessions
    naive_orch.trace_repo = traces

    async def _no_gap(*args, **kwargs):
        return {"chunk_count": 0, "embedded_count": 0, "index_exists": False}

    async def _empty_cosine(*args, **kwargs):
        return [], 0

    naive_orch._diagnose_corpus = _no_gap
    naive_orch._mongo_cosine = _empty_cosine

    monkeypatch.setattr(pipelines, "get_naive", lambda: naive_orch)
    monkeypatch.setattr(pipelines, "get_two_stage", lambda: two_stage_orch)
    monkeypatch.setattr(pipelines, "get_full", lambda: full_orch)
    monkeypatch.setattr(pipelines, "TraceRepository", lambda: traces)
    monkeypatch.setattr(pipelines, "SessionRepository", lambda: sessions)

    feedback_store: List[tuple[str, List[Dict[str, Any]]]] = []

    async def _record_feedback(service_id: str, documents: List[Dict[str, Any]]) -> None:
        feedback_store.append((service_id, documents))

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")
    app.include_router(embedding_endpoint.router, prefix="/api/v1/embedding")
    app.include_router(cds_router)
    async def _lookup_card(card_id: str):
        for (trace_id, _org_id), doc in traces.rows.items():
            if trace_id == card_id:
                return doc
        return None

    app.dependency_overrides[get_feedback_recorder] = lambda: _record_feedback
    app.dependency_overrides[get_card_lookup] = lambda: _lookup_card

    client = TestClient(app)
    clinician_token = _sign(
        secret,
        {"sub": "dr-patel", "org_id": "clinic-1", "role": "clinician", "exp": int(time.time()) + 300},
    )
    other_org_token = _sign(
        secret,
        {"sub": "dr-other", "org_id": "clinic-2", "role": "clinician", "exp": int(time.time()) + 300},
    )
    headers = {"Authorization": f"Bearer {clinician_token}"}

    # 1. Bedrock runtime readiness & embedding identity
    ready = client.get("/api/v1/pipelines/ready", headers=headers)
    assert ready.status_code == 200
    assert ready.json()["generation_backend"] == "bedrock"
    assert ready.json()["generation_model"] == "global.openai.gpt-6-luna"
    assert ready.json()["embedding_model"] == "global.cohere.embed-v4:0"
    assert ready.json()["reranker_model"] == "cohere.rerank-v3-5:0"

    embed_info = client.get("/api/v1/embedding/info", headers=headers)
    assert embed_info.status_code == 200
    assert embed_info.json()["device"] == "bedrock"
    assert embed_info.json()["dimension"] == 1536

    # 2. Full multi-agent control plane: RETRIEVE_MORE -> targeted safety loop -> ACCEPT -> synthesis
    full_resp = client.post(
        "/api/v1/pipelines/full/chat",
        headers=headers,
        json={
            "query": "First-line antibiotic for community pneumonia with allergy check",
            "user_id": "dr-patel",
            "org_id": "clinic-1",
            "patient_id": "patient-42",
        },
    )
    assert full_resp.status_code == 200
    full_data = full_resp.json()
    assert full_data["pipeline"] == "full"
    assert full_data["generator_profile"] == "bedrock"
    assert full_data["policy_decision"]["action"] == PolicyAction.ACCEPT.value
    assert full_data["policy_decision"]["passed"] is True
    assert len(full_retriever.calls) == 2
    assert full_data["answer"].startswith("Clinician decision support only:")
    assert "cpg-1" in full_data["answer_provenance"]["cited_chunk_ids"]
    assert "safety-1" in full_data["answer_provenance"]["cited_chunk_ids"]
    assert "fabricated-999" in full_data["answer_provenance"]["rejected_chunk_ids"]
    assert full_llms["safety"].calls  # Safety specialist agent actually ran on iteration 1
    assert full_llms["guideline"].calls  # Guideline specialist agent ran

    # 3. Audit trace & session reads are org-bound and include the full control-plane record
    trace_id = full_data["trace_id"]
    trace_resp = client.get(
        f"/api/v1/pipelines/traces/{trace_id}",
        headers=headers,
        params={"org_id": "clinic-1", "include_details": "true"},
    )
    assert trace_resp.status_code == 200
    trace_body = trace_resp.json()
    assert trace_body["sufficiency_checks_count"] == 2
    actions = [item["action"] for item in trace_body["policy_decisions"]]
    assert actions == [PolicyAction.RETRIEVE_MORE.value, PolicyAction.ACCEPT.value]

    session_id = trace_body["session_id"]
    session_resp = client.get(
        f"/api/v1/pipelines/sessions/{session_id}",
        headers=headers,
        params={"org_id": "clinic-1"},
    )
    assert session_resp.status_code == 200
    assert session_resp.json()["user_id"] == "dr-patel"

    # Cross-org trace read is blocked
    cross_org_trace = client.get(
        f"/api/v1/pipelines/traces/{trace_id}",
        headers={"Authorization": f"Bearer {other_org_token}"},
        params={"org_id": "clinic-1"},
    )
    assert cross_org_trace.status_code == 403

    # 4. Compare all three pipelines: naive abstains on empty retrieval without calling LLM
    compare_resp = client.post(
        "/api/v1/pipelines/compare",
        headers=headers,
        json={
            "query": "First-line antibiotic for community pneumonia with allergy check",
            "user_id": "dr-patel",
            "org_id": "clinic-1",
        },
    )
    assert compare_resp.status_code == 200
    compare_body = compare_resp.json()
    assert set(compare_body["results"].keys()) == {"naive", "two-stage", "full"}
    assert compare_body["results"]["naive"]["policy_decision"]["action"] == PolicyAction.INSUFFICIENT_EVIDENCE.value
    assert compare_body["results"]["naive"]["answer"].startswith("Clinician decision support only:")
    assert naive_llm.calls == []
    assert compare_body["results"]["two-stage"]["policy_decision"]["action"] == PolicyAction.ACCEPT.value
    assert compare_body["results"]["full"]["policy_decision"]["action"] == PolicyAction.ACCEPT.value

    # 5. CDS Hooks discovery, accepted card summary, and authenticated override feedback
    discovery = client.get("/cds-services", headers=headers)
    assert discovery.status_code == 200
    assert discovery.json()["services"][0]["id"] == "medswin-evidence"

    card_resp = client.post(
        "/cds-services/medswin-evidence",
        headers=headers,
        json={
            "hook": "patient-view",
            "hookInstance": "hook-e2e-1",
            "context": {
                "question": "First-line antibiotic for community pneumonia with allergy check",
                "patientId": "patient-42",
                "userId": "dr-patel",
                "orgId": "clinic-1",
            },
        },
    )
    assert card_resp.status_code == 200
    card = card_resp.json()["cards"][0]
    assert card["indicator"] == "info"
    assert card["summary"].startswith("Amoxicillin is first-line")
    assert "Clinician decision support only" not in card["summary"]
    assert "does not establish a diagnosis" in card["detail"]

    fb_resp = client.post(
        "/cds-services/medswin-evidence/feedback",
        headers=headers,
        json={
            "feedback": [
                {
                    "card": card["uuid"],
                    "outcome": "overridden",
                    "overrideReason": {
                        "reason": {"code": "allergy"},
                        "userComment": "Patient MRN: 998877 has severe allergy.",
                    },
                }
            ]
        },
    )
    assert fb_resp.status_code == 200
    doc = feedback_store[0][1][0]
    assert doc["user_id"] == "dr-patel"
    assert doc["org_id"] == "clinic-1"
    assert "998877" not in doc["override_comment"]


def test_full_pipeline_skips_specialist_agents_when_retrieval_is_empty():
    import asyncio

    sessions = _MemorySessions()
    traces = _MemoryTraces()
    empty_orch, llms, _ = _wire_orchestrator(PipelineId.FULL, sessions, traces, empty=True)
    response = asyncio.run(
        empty_orch.chat(
            query="First-line antibiotic for pneumonia",
            user_id="dr-patel",
            org_id="clinic-1",
        )
    )
    assert response.policy_decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert response.answer_artifact.abstained is True
    for role, llm in llms.items():
        assert llm.calls == [], f"{role} LLM should not run when retrieval returns no passages"
