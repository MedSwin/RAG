"""Bind a request to the authenticated organisation."""

from __future__ import annotations

from fastapi import HTTPException, Request


def fhir_id(value: str) -> str:
    """Return the id tail of a FHIR reference such as Patient/pt-1."""
    text = (value or "").strip()
    if "/" in text:
        return text.rsplit("/", 1)[-1].strip()
    return text


def bind_tenant(request: Request, user_id: str, org_id: str) -> tuple[str, str]:
    """Return the caller identity. A token wins over a different body value."""
    supplied_user = fhir_id(user_id)
    supplied_org = org_id.strip()
    token_user = fhir_id(str(getattr(request.state, "user_id", "") or ""))
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
    """Return the organisation for a read or write. A mismatch returns 403."""
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    _, bound_org = bind_tenant(request, token_user or "anonymous", org_id)
    return bound_org


def require_operator(request: Request) -> None:
    """Allow a local caller. A signed caller must hold the operator role."""
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    if not token_user:
        return
    roles = getattr(request.state, "roles", []) or []
    held = {str(role).strip().lower() for role in roles}
    if "operator" not in held:
        raise HTTPException(status_code=403, detail="An operator role is required")


def bind_body_org(request: Request, body):
    """Return the body with the token organisation. A mismatch returns 403."""
    org_id = bind_org(request, str(body.org_id))
    if org_id == body.org_id:
        return body
    return body.model_copy(update={"org_id": org_id})
