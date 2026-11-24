import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_auth import _token

from app.api.auth import AuthMiddleware
from app.api.cds import get_card_lookup, get_cds_chat, get_feedback_recorder, router
from app.core.config import settings
from app.medswin.cds import cards_from_chat, feedback_documents
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


def test_missing_question_returns_a_warning_card_and_does_not_call_chat():
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        raise AssertionError("chat should not run")

    response = _app(chat).post(
        "/cds-services/medswin-evidence",
        json={"hook": "patient-view", "hookInstance": "hook-1", "context": {"patientId": "p1"}},
    )
    assert response.status_code == 200
    card = response.json()["cards"][0]
    assert card["indicator"] == "warning"
    assert "No clinical question" in card["summary"]
    assert "p1" in card["detail"]
    assert calls == []


def test_card_summary_skips_the_decision_support_boundary():
    response = _response(accepted=True).model_copy(
        update={
            "answer": (
                "Clinician decision support only: this response supports clinical review and "
                "does not establish a final diagnosis. Amoxicillin is listed in the cited guideline passage."
            )
        }
    )
    card = cards_from_chat(response)[0]
    assert card["summary"].startswith("Amoxicillin is listed")
    assert "Clinician decision support only" not in card["summary"]
    assert "does not establish a diagnosis" in card["detail"]


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


def test_a_crashed_check_returns_a_critical_card():
    response = _response(accepted=False).model_copy(
        update={
            "policy_decision": None,
            "degraded_mode": {"error": True},
            "trace_id": "trace-crash",
        }
    )
    card = cards_from_chat(response, hook_instance="hook-9", patient_id="pt-1")[0]
    assert card["indicator"] == "critical"
    assert card["uuid"] == "trace-crash"
    assert card["summary"] == "MedSwin could not complete the evidence check."
    assert card["extension"] == {
        "trace_id": "trace-crash",
        "hookInstance": "hook-9",
        "patientId": "pt-1",
    }


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


def test_cds_discovery_stays_open_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    response = TestClient(app).get("/cds-services")
    assert response.status_code == 200
    assert "medswin-evidence" in {item["id"] for item in response.json()["services"]}


def test_cds_service_accepts_a_fhir_user_reference(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", "clinical-secret-value")
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    seen = {}

    async def chat(**kwargs):
        seen.update(kwargs)
        return _response(accepted=True)

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_cds_chat] = lambda: chat
    now = int(time.time())
    token = _token(
        "clinical-secret-value",
        {"sub": "dr-chen", "org_id": "st-vincent-cdss", "role": "clinician", "exp": now + 60},
    )
    response = TestClient(app).post(
        "/cds-services/medswin-evidence",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "hook": "patient-view",
            "hookInstance": "hook-1",
            "context": {
                "userId": "Practitioner/dr-chen",
                "patientId": "Patient/pt-1",
                "orgId": "st-vincent-cdss",
                "clinicalQuestion": "Which anticoagulant is safer?",
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["cards"]
    assert seen["user_id"] == "dr-chen"
    assert seen["patient_id"] == "pt-1"
    assert seen["question"] == "Which anticoagulant is safer?"


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


def test_feedback_requires_a_signed_caller():
    recorded = []

    async def record(service_id, documents):
        recorded.append((service_id, documents))

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_feedback_recorder] = lambda: record
    response = TestClient(app).post(
        "/cds-services/medswin-evidence/feedback",
        json={
            "feedback": [
                {
                    "card": "card-1",
                    "outcome": "overridden",
                    "acceptedSuggestions": [{"id": "sug-1"}],
                    "overrideReason": {
                        "reason": {"code": "clinical-judgement"},
                        "userComment": "The local protocol differs.",
                    },
                    "outcomeTimestamp": "2026-11-24T00:00:00Z",
                }
            ]
        },
    )
    assert response.status_code == 401
    assert recorded == []


def test_feedback_rejects_an_unknown_outcome():
    response = _app(None).post(
        "/cds-services/medswin-evidence/feedback",
        json={"feedback": [{"card": "card-1", "outcome": "ignored"}]},
    )
    assert response.status_code == 422


def test_feedback_unknown_service_is_not_found():
    response = _app(None).post(
        "/cds-services/other/feedback",
        json={"feedback": [{"card": "card-1", "outcome": "accepted"}]},
    )
    assert response.status_code == 404


def test_feedback_document_keeps_the_audit_fields():
    from app.api.cds import FeedbackItem

    item = FeedbackItem(card="card-9", outcome="accepted")
    documents = feedback_documents("medswin-evidence", [item], "2026-11-24T00:00:00+00:00")
    assert documents == [
        {
            "service_id": "medswin-evidence",
            "card": "card-9",
            "outcome": "accepted",
            "accepted_suggestions": [],
            "override_code": None,
            "override_comment": None,
            "outcome_timestamp": None,
            "recorded_at": "2026-11-24T00:00:00+00:00",
        }
    ]


def test_authenticated_feedback_binds_caller_and_redacts_phi(monkeypatch):
    import base64
    import hashlib
    import hmac
    import json
    import time

    def _sign(secret: str, payload: dict) -> str:
        def _enc(obj):
            return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

        head = _enc({"alg": "HS256", "typ": "JWT"})
        body = _enc(payload)
        sig = hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
        return f"{head}.{body}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"

    secret = "cds-audit-secret"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    recorded = []

    async def record(service_id, documents):
        recorded.append((service_id, documents))

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_feedback_recorder] = lambda: record

    async def lookup(card_id: str):
        if card_id == "card-2":
            return {"trace_id": card_id, "user_id": "clinician-7", "org_id": "org-a"}
        return None

    app.dependency_overrides[get_card_lookup] = lambda: lookup
    client = TestClient(app)

    good = _sign(secret, {"sub": "clinician-7", "org_id": "org-a", "exp": int(time.time()) + 60})
    response = client.post(
        "/cds-services/medswin-evidence/feedback",
        headers={"Authorization": f"Bearer {good}"},
        json={
            "feedback": [
                {
                    "card": "card-2",
                    "outcome": "overridden",
                    "overrideReason": {
                        "reason": {"code": "patient-allergy"},
                        "userComment": "Patient MRN: 123456 reported rash; email jane@example.com.",
                    },
                }
            ]
        },
    )
    assert response.status_code == 200
    doc = recorded[0][1][0]
    assert doc["user_id"] == "clinician-7"
    assert doc["org_id"] == "org-a"
    assert "123456" not in doc["override_comment"]
    assert "jane@example.com" not in doc["override_comment"]
    assert "[REDACTED_PATIENT_ID]" in doc["override_comment"]
    assert "[REDACTED_EMAIL]" in doc["override_comment"]

    foreign = client.post(
        "/cds-services/medswin-evidence/feedback",
        headers={"Authorization": f"Bearer {good}"},
        json={"feedback": [{"card": "card-foreign", "outcome": "accepted"}]},
    )
    assert foreign.status_code == 403
    assert len(recorded) == 1

    no_org = _sign(secret, {"sub": "clinician-7", "exp": int(time.time()) + 60})
    denied = client.post(
        "/cds-services/medswin-evidence/feedback",
        headers={"Authorization": f"Bearer {no_org}"},
        json={"feedback": [{"card": "card-3", "outcome": "accepted"}]},
    )
    assert denied.status_code == 403
    assert len(recorded) == 1

    wrong_org = client.post(
        "/cds-services/medswin-evidence",
        headers={"Authorization": f"Bearer {good}"},
        json={"hook": "patient-view", "hookInstance": "hook-2", "context": {"orgId": "org-b"}},
    )
    assert wrong_org.status_code == 403
