from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import AuthMiddleware
from app.api.cds import get_cds_chat, router
from app.core.config import settings
from app.medswin.cds import cards_from_chat
from app.schemas.enums import PolicyAction, SourceType
from app.schemas.evidence import AnswerArtifact, CandidatePassage, EvidenceBundle, PolicyDecision
from app.schemas.traces import ChatResponse


def _response(*, accepted: bool, unresolved: bool = False) -> ChatResponse:
    passage = CandidatePassage(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_type=SourceType.CPG,
        text="Guideline text.",
    )
    decision = PolicyDecision(
        passed=accepted and not unresolved,
        action=PolicyAction.ACCEPT if accepted and not unresolved else PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="The gate recorded this reason.",
        missing_facets=[] if accepted else ["safety_contraindications"],
        unresolved_critical_conflicts=unresolved,
    )
    return ChatResponse(
        answer="Amoxicillin is listed in the cited guideline passage.",
        answer_artifact=AnswerArtifact(
            content="Amoxicillin is listed in the cited guideline passage.",
            rendered_text="Amoxicillin is listed in the cited guideline passage.",
            abstained=not accepted or unresolved,
        ),
        evidence_bundle=EvidenceBundle(
            passages=[passage],
            total_tokens=12,
            cpg_count=1,
            emr_count=0,
            lit_count=0,
        ),
        trace_id="trace-1",
        policy_decision=decision,
    )


def _app(chat):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_cds_chat] = lambda: chat
    return TestClient(app)


def test_discovery_names_the_evidence_service():
    response = _app(None).get("/cds-services")
    assert response.status_code == 200
    service = response.json()["services"][0]
    assert service["id"] == "medswin-evidence"
    assert service["hook"] == "patient-view"


def test_missing_question_returns_no_cards_and_does_not_call_chat():
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        raise AssertionError("chat should not run")

    response = _app(chat).post(
        "/cds-services/medswin-evidence",
        json={"hook": "patient-view", "hookInstance": "hook-1", "context": {"patientId": "p1"}},
    )
    assert response.status_code == 200
    assert response.json() == {"cards": []}
    assert calls == []


def test_accepted_decision_returns_an_info_card():
    async def chat(**kwargs):
        assert kwargs["question"] == "What is the first-line antibiotic?"
        assert kwargs["user_id"] == "clinician-1"
        assert kwargs["org_id"] == "demo-org"
        assert kwargs["patient_id"] == "patient-1"
        return _response(accepted=True)

    response = _app(chat).post(
        "/cds-services/medswin-evidence",
        json={
            "hook": "patient-view",
            "hookInstance": "hook-1",
            "context": {
                "question": "What is the first-line antibiotic?",
                "userId": "clinician-1",
                "orgId": "demo-org",
                "patientId": "patient-1",
            },
        },
    )
    assert response.status_code == 200
    card = response.json()["cards"][0]
    assert card["indicator"] == "info"
    assert card["source"]["label"] == "MedSwin"
    assert len(card["summary"]) <= 140
    assert "chunk-1" in card["detail"]
    assert "Clinician decision support only" in card["detail"]


def test_insufficient_decision_returns_a_warning_card():
    cards = cards_from_chat(_response(accepted=False))
    assert cards[0]["indicator"] == "warning"
    assert cards[0]["summary"] == "The available evidence is insufficient."
    assert "safety_contraindications" in cards[0]["detail"]


def test_unresolved_conflict_returns_a_critical_card():
    cards = cards_from_chat(_response(accepted=True, unresolved=True))
    assert cards[0]["indicator"] == "critical"


def test_unknown_service_is_not_found():
    response = _app(None).post(
        "/cds-services/other",
        json={"hook": "patient-view", "hookInstance": "hook-1", "context": {}},
    )
    assert response.status_code == 404


def test_cds_service_requires_a_token_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    response = TestClient(app).post(
        "/cds-services/medswin-evidence",
        json={"hook": "patient-view", "hookInstance": "hook-1", "context": {}},
    )
    assert response.status_code == 401
