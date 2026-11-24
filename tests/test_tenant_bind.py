import base64
import hashlib
import hmac
import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import AuthMiddleware
from app.api.cds import get_cds_chat, router as cds_router
from app.api.tenant import bind_org, bind_tenant
from app.api.v1.endpoints import pipelines, retrieval, storage
from app.core.config import settings
from app.schemas.enums import PipelineId
from app.schemas.evidence import EvidenceBundle
from app.schemas.traces import ChatResponse


class _State:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _Request:
    def __init__(self, **state):
        self.state = _State(**state)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _token(secret: str, claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(signature)}"


def test_unsigned_request_keeps_the_body_organisation():
    user_id, org_id = bind_tenant(_Request(), "clinician-1", "org-a")
    assert (user_id, org_id) == ("clinician-1", "org-a")
    assert bind_org(_Request(), "org-a") == "org-a"


def test_token_rejects_a_different_organisation():
    request = _Request(user_id="clinician-1", org_id="org-a")
    try:
        bind_tenant(request, "clinician-1", "org-b")
    except Exception as exc:
        assert exc.status_code == 403
        assert "organisation" in exc.detail
    else:
        raise AssertionError("A different organisation was accepted")


def test_token_fills_a_missing_organisation():
    request = _Request(user_id="clinician-1", org_id="org-a")
    assert bind_tenant(request, "", "") == ("clinician-1", "org-a")


def test_cds_token_blocks_another_organisation(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        raise AssertionError("chat should not run")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cds_router)
    app.dependency_overrides[get_cds_chat] = lambda: chat
    token = _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})
    response = TestClient(app).post(
        "/cds-services/medswin-evidence",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "hook": "patient-view",
            "hookInstance": "hook-1",
            "context": {"question": "What is the first-line antibiotic?", "orgId": "org-b", "userId": "clinician-1"},
        },
    )
    assert response.status_code == 403
    assert calls == []


def test_pipeline_token_uses_the_token_organisation(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    seen = []

    async def _chat(pipeline, request):
        seen.append((pipeline, request.user_id, request.org_id))
        return ChatResponse(
            answer="held",
            evidence_bundle=EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0),
            trace_id="trace-1",
            pipeline=pipeline,
        )

    monkeypatch.setattr(pipelines, "_chat", _chat)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")
    token = _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})
    response = TestClient(app).post(
        "/api/v1/pipelines/two-stage/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "q", "user_id": "clinician-1", "org_id": "org-a"},
    )
    assert response.status_code == 200
    assert seen == [(PipelineId.TWO_STAGE, "clinician-1", "org-a")]

    denied = TestClient(app).post(
        "/api/v1/pipelines/two-stage/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "q", "user_id": "clinician-1", "org_id": "org-b"},
    )
    assert denied.status_code == 403
    assert seen == [(PipelineId.TWO_STAGE, "clinician-1", "org-a")]


def _auth(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    return _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})


def test_storage_token_blocks_another_organisation(monkeypatch):
    token = _auth(monkeypatch)
    listed = []

    class Storage:
        async def list_chunks(self, skip, limit, filters):
            listed.append(filters["org_id"])
            return []

        async def clear_benchmark_org(self, org_id, remove_indexes):
            raise AssertionError("reset should not run")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    denied = client.get("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-b"})
    assert denied.status_code == 403
    assert listed == []
    allowed = client.get("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-a"})
    assert allowed.status_code == 200
    assert listed == ["org-a"]
    reset = client.post(
        "/api/v1/storage/benchmark/reset",
        headers=headers,
        json={"org_id": "org-b"},
    )
    assert reset.status_code == 403


def test_retrieval_token_blocks_another_organisation_before_embed(monkeypatch):
    token = _auth(monkeypatch)

    async def _search(request):
        raise AssertionError("search should not run")

    monkeypatch.setattr(retrieval, "_search", _search)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(retrieval.router, prefix="/api/v1/retrieval")
    response = TestClient(app).post(
        "/api/v1/retrieval/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "first-line antibiotic", "org_id": "org-b", "use_reranking": False},
    )
    assert response.status_code == 403
