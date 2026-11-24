"""Bearer JWT gate. It runs only when ENABLE_AUTH is true."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Callable, Dict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)

_PUBLIC_PREFIXES = ("/health", "/docs", "/openapi", "/redoc", "/app")


def _is_public_path(path: str) -> bool:
    """Match a public route on a full path segment."""
    if path == "/":
        return True
    return any(path == prefix or path.startswith(prefix + "/") for prefix in _PUBLIC_PREFIXES)


def _claim_roles(claims: Dict[str, Any]) -> set[str]:
    found: set[str] = set()
    role = claims.get("role")
    if isinstance(role, str) and role.strip():
        found.add(role.strip().lower())
    roles = claims.get("roles")
    if isinstance(roles, list):
        for item in roles:
            if isinstance(item, str) and item.strip():
                found.add(item.strip().lower())
    return found


def _allowed_roles() -> set[str]:
    return {
        item.strip().lower()
        for item in (settings.RBAC_ALLOWED_ROLES or "").split(",")
        if item.strip()
    }


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _audience_matches(claims: Dict[str, Any], expected: str) -> bool:
    audience = claims.get("aud")
    if isinstance(audience, str):
        return audience.strip() == expected
    if isinstance(audience, list):
        return any(isinstance(item, str) and item.strip() == expected for item in audience)
    return False


def _issuer_matches(claims: Dict[str, Any], expected: str) -> bool:
    issuer = claims.get("iss")
    return isinstance(issuer, str) and issuer.strip() == expected


def verify_hs256(token: str, secret: str, *, now: float | None = None) -> Dict[str, Any]:
    """Verify one HS256 JWT. Reject every other algorithm, including none."""
    parts = token.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("malformed")
    try:
        header = json.loads(_b64url_decode(parts[0]))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("header") from exc
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise ValueError("algorithm")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        actual = _b64url_decode(parts[2])
    except ValueError as exc:
        raise ValueError("signature") from exc
    if len(actual) != len(expected) or not hmac.compare_digest(actual, expected):
        raise ValueError("signature")
    try:
        claims = json.loads(_b64url_decode(parts[1]))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("claims") from exc
    if not isinstance(claims, dict):
        raise ValueError("claims")
    current = time.time() if now is None else now
    expires = claims.get("exp")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        raise ValueError("exp")
    expires = float(expires)
    if current >= expires:
        raise ValueError("expired")
    if expires - current > 7 * 24 * 3600:
        raise ValueError("exp")
    not_before = claims.get("nbf")
    if not_before is not None:
        if isinstance(not_before, bool) or not isinstance(not_before, (int, float)):
            raise ValueError("not_before")
        if current < float(not_before):
            raise ValueError("not_before")
    expected_audience = (settings.AUTH_JWT_AUDIENCE or "").strip()
    if expected_audience and not _audience_matches(claims, expected_audience):
        raise ValueError("audience")
    expected_issuer = (settings.AUTH_JWT_ISSUER or "").strip()
    if expected_issuer and not _issuer_matches(claims, expected_issuer):
        raise ValueError("issuer")
    return claims


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a signed HS256 bearer token when ENABLE_AUTH is true."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if _is_public_path(request.url.path):
            return await call_next(request)
        # CDS Hooks discovery is a catalog. Service calls stay behind the token.
        if request.method == "GET" and request.url.path.rstrip("/") == "/cds-services":
            return await call_next(request)
        if settings.ENABLE_RBAC and not settings.ENABLE_AUTH:
            return JSONResponse(status_code=401, content={"detail": "Authentication is not configured"})
        if not settings.ENABLE_AUTH:
            return await call_next(request)
        header = request.headers.get("Authorization") or ""
        if not header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing Bearer token"})
        token = header[7:].strip()
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Empty Bearer token"})
        secret = (settings.AUTH_JWT_SECRET or "").strip()
        if not secret:
            return JSONResponse(status_code=401, content={"detail": "Authentication is not configured"})
        algorithm = (settings.AUTH_JWT_ALGORITHM or "").strip().upper()
        if algorithm != "HS256":
            return JSONResponse(status_code=401, content={"detail": "Unsupported JWT algorithm"})
        try:
            claims = verify_hs256(token, secret)
        except (ValueError, OverflowError, TypeError):
            return JSONResponse(status_code=401, content={"detail": "Invalid Bearer token"})
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            return JSONResponse(status_code=401, content={"detail": "Invalid Bearer token"})
        request.state.user_id = subject
        org = claims.get("org_id") if claims.get("org_id") is not None else claims.get("org")
        request.state.org_id = str(org).strip() if org else None
        roles = _claim_roles(claims)
        request.state.roles = sorted(roles)
        clinical = bool(roles.intersection(_allowed_roles()))
        if settings.ENABLE_RBAC and not clinical and "operator" not in roles:
            return JSONResponse(status_code=403, content={"detail": "Clinical role is required"})
        return await call_next(request)
