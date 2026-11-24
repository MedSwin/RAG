"""Bind a request to the authenticated organisation."""

from __future__ import annotations

from fastapi import HTTPException, Request


def bind_tenant(request: Request, user_id: str, org_id: str) -> tuple[str, str]:
    """Return the caller identity. A token wins over a different body value."""
    supplied_user = user_id.strip()
    supplied_org = org_id.strip()
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    if not token_user:
        return supplied_user, supplied_org
    token_org = str(getattr(request.state, "org_id", "") or "").strip()
    if not token_org:
        raise HTTPException(status_code=403, detail="The token has no organisation")
    if supplied_user and supplied_user != token_user:
        raise HTTPException(status_code=403, detail="The token subject does not match the request user")
    if supplied_org and supplied_org != token_org:
        raise HTTPException(status_code=403, detail="The token organisation does not match the request organisation")
    return token_user, token_org


def bind_org(request: Request, org_id: str) -> str:
    """Return the organisation for a read. A token org replaces a different query value."""
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    _, bound_org = bind_tenant(request, token_user or "anonymous", org_id)
    return bound_org
