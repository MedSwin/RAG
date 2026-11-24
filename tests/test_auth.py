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
    assert response.json() == {"user_id": "clinician-1", "org_id": "demo-org"}


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


def test_health_stays_open_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", None)
    response = _client().get("/health")
    assert response.status_code == 200
