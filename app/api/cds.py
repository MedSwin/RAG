"""CDS Hooks discovery and the MedSwin evidence service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.tenant import bind_tenant
from app.medswin.cds import SERVICE_ID, cards_from_chat, discovery, feedback_documents, missing_question_card

router = APIRouter()

ChatFunc = Callable[..., Awaitable[Any]]


class CdsContext(BaseModel):
    model_config = ConfigDict(extra="allow")
    patientId: Optional[str] = None
    userId: Optional[str] = None
    orgId: Optional[str] = None
    question: Optional[str] = None


class CdsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    hook: str
    hookInstance: str = Field(min_length=1)
    context: CdsContext = Field(default_factory=CdsContext)


async def default_cds_chat(
    question: str,
    user_id: str,
    org_id: str,
    patient_id: Optional[str],
):
    """Run the two-stage gate. Tests replace this dependency."""
    from app.api.v1.endpoints.pipelines import get_two_stage

    return await get_two_stage().chat(
        query=question,
        user_id=user_id,
        org_id=org_id,
        patient_id=patient_id,
        constraints={},
    )


def get_cds_chat() -> ChatFunc:
    return default_cds_chat


class OverrideReason(BaseModel):
    model_config = ConfigDict(extra="allow")
    reason: Optional[Dict[str, Any]] = None
    userComment: Optional[str] = None


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    card: str = Field(min_length=1)
    outcome: Literal["accepted", "overridden"]
    acceptedSuggestions: List[Dict[str, Any]] = Field(default_factory=list)
    overrideReason: Optional[OverrideReason] = None
    outcomeTimestamp: Optional[str] = None


class FeedbackRequest(BaseModel):
    feedback: List[FeedbackItem] = Field(min_length=1)


FeedbackFunc = Callable[[str, List[Dict[str, Any]]], Awaitable[None]]


async def default_record_feedback(service_id: str, documents: List[Dict[str, Any]]) -> None:
    """Store CDS feedback. This path does not call a model."""
    from app.core.database import get_database

    await get_database().cds_feedback.insert_many(documents)


def get_feedback_recorder() -> FeedbackFunc:
    return default_record_feedback


@router.get("/cds-services")
async def cds_discovery() -> Dict[str, Any]:
    return discovery()


def _fhir_id(value: str) -> str:
    """Return the id tail of a FHIR reference such as Practitioner/dr-chen."""
    text = (value or "").strip()
    if "/" in text:
        return text.rsplit("/", 1)[-1].strip()
    return text


def _cds_caller(request: Request, body: CdsRequest) -> tuple[str, str]:
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    if token_user:
        token_org = str(getattr(request.state, "org_id", "") or "").strip()
        return bind_tenant(
            request,
            _fhir_id(body.context.userId or token_user),
            (body.context.orgId or token_org).strip(),
        )
    user_id = (body.context.userId or "cds-client").strip() or "cds-client"
    org_id = (body.context.orgId or "cds").strip() or "cds"
    return user_id, org_id


@router.post("/cds-services/{service_id}")
async def cds_service(
    service_id: str,
    body: CdsRequest,
    request: Request,
    chat: ChatFunc = Depends(get_cds_chat),
) -> Dict[str, Any]:
    if service_id != SERVICE_ID:
        raise HTTPException(status_code=404, detail="Unknown CDS service")
    if body.hook != "patient-view":
        raise HTTPException(status_code=400, detail="This service expects the patient-view hook")
    user_id, org_id = _cds_caller(request, body)
    extra = body.context.model_extra or {}
    question = " ".join((body.context.question or str(extra.get("clinicalQuestion") or "")).split())
    if not question:
        return {"cards": [missing_question_card(body.context.patientId)]}
    response = await chat(
        question=question,
        user_id=user_id,
        org_id=org_id,
        patient_id=(body.context.patientId or "").strip() or None,
    )
    return {"cards": cards_from_chat(response)}


@router.post("/cds-services/{service_id}/feedback")
async def cds_feedback(
    service_id: str,
    body: FeedbackRequest,
    request: Request,
    record: FeedbackFunc = Depends(get_feedback_recorder),
) -> Dict[str, Any]:
    if service_id != SERVICE_ID:
        raise HTTPException(status_code=404, detail="Unknown CDS service")
    token_user = str(getattr(request.state, "user_id", "") or "").strip()
    user_id = None
    org_id = None
    if token_user:
        token_org = str(getattr(request.state, "org_id", "") or "").strip()
        user_id, org_id = bind_tenant(request, token_user, token_org)
    recorded_at = datetime.now(timezone.utc).isoformat()
    documents = feedback_documents(
        service_id,
        body.feedback,
        recorded_at,
        user_id=user_id,
        org_id=org_id,
    )
    await record(service_id, documents)
    return {}
