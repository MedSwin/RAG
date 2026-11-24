"""CDS Hooks discovery and the MedSwin evidence service."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.medswin.cds import SERVICE_ID, cards_from_chat, discovery

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


@router.get("/cds-services")
async def cds_discovery() -> Dict[str, Any]:
    return discovery()


@router.post("/cds-services/{service_id}")
async def cds_service(
    service_id: str,
    body: CdsRequest,
    chat: ChatFunc = Depends(get_cds_chat),
) -> Dict[str, Any]:
    if service_id != SERVICE_ID:
        raise HTTPException(status_code=404, detail="Unknown CDS service")
    if body.hook != "patient-view":
        raise HTTPException(status_code=400, detail="This service expects the patient-view hook")
    question = " ".join((body.context.question or "").split())
    if not question:
        return {"cards": []}
    response = await chat(
        question=question,
        user_id=(body.context.userId or "cds-client").strip() or "cds-client",
        org_id=(body.context.orgId or "cds").strip() or "cds",
        patient_id=(body.context.patientId or "").strip() or None,
    )
    return {"cards": cards_from_chat(response)}
