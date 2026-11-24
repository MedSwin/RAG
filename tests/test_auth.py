import base64
import hashlib
import hmac
import json
import time

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.auth import AuthMiddleware
from app.core.config import settings


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _token(secret: str, claims: dict, *, algorithm: str = "HS256") -> str:
    header = _b64url(json.dumps({"alg": algorithm, "typ": "JWT"}).encode())
    body = _b64url(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(signature)}"


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/api/v1/ping")
    def ping(request: Request):
        return {
            "user_id": getattr(request.state, "user_id", None),
            "org_id": getattr(request.state, "org_id", None),
            "roles": list(getattr(request.state, "roles", []) or []),
        }

    @app.get("/health")
    def health():
        return {"ok": True}

    return TestClient(app)


def test_auth_off_allows_a_request_without_a_token(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    response = _client().get("/api/v1/ping")
    assert response.status_code == 200


def test_auth_on_rejects_a_missing_secret(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    response = _client().get("/api/v1/ping", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication is not configured"


def test_auth_on_rejects_a_prefix_that_is_not_a_signature(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    response = _client().get(
        "/api/v1/ping",
        headers={"Authorization": f"Bearer {secret[:8]}-still-not-signed"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid Bearer token"


def test_auth_on_accepts_a_signed_token(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    token = _token(secret, {"sub": "clinician-1", "org_id": "demo-org", "exp": int(time.time()) + 60})
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"user_id": "clinician-1", "org_id": "demo-org", "roles": []}


def test_cloud_mode_refuses_to_start_without_auth(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    try:
        settings.enforce_cloud_auth()
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_local_mode_can_start_without_auth(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    settings.enforce_cloud_auth()


def test_auth_on_rejects_a_token_with_no_expiry(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    token = _token(secret, {"sub": "clinician-1", "org_id": "demo-org"})
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_auth_on_rejects_a_token_expiry_that_is_not_a_number(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    token = _token(secret, {"sub": "clinician-1", "exp": ["later"]})
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_auth_on_rejects_an_expired_token(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    token = _token(secret, {"sub": "clinician-1", "exp": int(time.time()) - 10})
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_auth_on_rejects_the_none_algorithm(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = _b64url(json.dumps({"sub": "clinician-1"}).encode())
    token = f"{header}.{body}."
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def _enable_rbac(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", "clinical-secret-value")
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "RBAC_ALLOWED_ROLES", "clinician")


def test_rbac_accepts_a_clinician_role(monkeypatch):
    _enable_rbac(monkeypatch)
    token = _token(
        "clinical-secret-value",
        {"sub": "clinician-1", "org_id": "demo-org", "role": "clinician", "exp": int(time.time()) + 60},
    )
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["roles"] == ["clinician"]


def test_rbac_rejects_a_token_with_no_clinical_role(monkeypatch):
    _enable_rbac(monkeypatch)
    token = _token(
        "clinical-secret-value",
        {"sub": "clinician-1", "role": "patient", "exp": int(time.time()) + 60},
    )
    response = _client().get("/api/v1/ping", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Clinical role is required"


def test_rbac_without_auth_rejects_the_request(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    monkeypatch.setattr(settings, "ENABLE_RBAC", True)
    response = _client().get("/api/v1/ping")
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication is not configured"


def test_a_path_that_only_starts_with_health_requires_a_token(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", "clinical-secret-value")
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/healthcare")
    def healthcare():
        return {"ok": True}

    response = TestClient(app).get("/healthcare")
    assert response.status_code == 401


def test_public_health_hides_the_model_names(monkeypatch):
    from app.main import _visible_health

    full = {
        "status": "healthy",
        "database": "connected",
        "generation_model": "named-model",
        "embedding_model": "named-embedder",
        "index_provenance_error": "dimension mismatch",
    }
    monkeypatch.setattr(settings, "SYSTEM_EVAL_MODE", False)
    assert _visible_health(full) == {"status": "healthy", "database": "connected"}
    monkeypatch.setattr(settings, "SYSTEM_EVAL_MODE", True)
    assert _visible_health(full)["generation_model"] == "named-model"


def test_health_stays_open_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    response = _client().get("/health")
    assert response.status_code == 200


def test_clinician_shell_stays_open_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/")
    def root():
        return {"ok": True}

    @app.get("/app/")
    def app_shell():
        return {"ok": True}

    @app.get("/api/v1/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/app/").status_code == 200
    assert client.get("/api/v1/ping").status_code == 401
