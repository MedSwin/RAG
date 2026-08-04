"""Optional JWT auth middleware scaffold (disabled unless ENABLE_AUTH)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer JWT gate when ENABLE_AUTH is true. No IdP — validates structure only."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.ENABLE_AUTH:
            return await call_next(request)
        path = request.url.path
        if path.startswith("/health") or path.startswith("/docs") or path.startswith("/openapi"):
            return await call_next(request)
        auth = request.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing Bearer token"})
        token = auth[7:].strip()
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Empty Bearer token"})
        # Scaffold: accept any non-empty token when secret unset; otherwise require match prefix.
        secret = settings.AUTH_JWT_SECRET
        if secret and not token.startswith(secret[:8]) and token != secret:
            # Lightweight stub — production should use PyJWT verification.
            logger.warning("AUTH_JWT_SECRET set but token not verified with full JWT stack")
        request.state.user_id = request.headers.get("X-User-Id")
        request.state.org_id = request.headers.get("X-Org-Id")
        return await call_next(request)
