"""Naive-RAG baseline and single-query comparison endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1.endpoints.medswin import ChatRequest, get_orchestrator
from app.core.config import settings
from app.medswin.naive import NaiveRAGOrchestrator, compare_responses
from app.medswin.orchestrator import MedSwinOrchestrator
from app.schemas import ChatResponse
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter()

_naive: Optional[NaiveRAGOrchestrator] = None


class NaiveChatRequest(ChatRequest):
    """Same contract as MedSwin chat, plus an optional dense top-K override."""

    top_k: Optional[int] = Field(None, description="Dense top-K. Defaults to NAIVE_TOP_K.")


class CompareResponse(BaseModel):
    query: str
    naive: ChatResponse
    medswin: ChatResponse
    diff: Dict[str, Any]


def get_naive_orchestrator() -> NaiveRAGOrchestrator:
    global _naive
    if _naive is None:
        cloud_model = settings.CLOUD_MODEL if settings.CLOUD_MODE else None
        _naive = NaiveRAGOrchestrator(
            embedding_client=EmbeddingClient(settings.active_embedding_url()),
            llm_client=LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model),
        )
    return _naive


@router.get("/ready")
async def naive_ready():
    """Lightweight preflight for the services naive-RAG actually calls."""
    from pathlib import Path

    from app.core.database import get_database

    status: Dict[str, Any] = {
        "pipeline": "naive_rag",
        "cloud_mode": settings.CLOUD_MODE,
        "embedding_url": settings.active_embedding_url(),
        "llm_url": settings.active_llm_url(settings.SUPERVISOR_URL),
        "index_exists": Path(settings.HNSW_INDEX_PATH).exists(),
        "mongo": False,
        "ready": False,
    }
    try:
        db = get_database()
        await db.command("ping")
        status["mongo"] = True
        status["chunk_count"] = await db.chunks.count_documents({})
        status["embedded_count"] = await db.chunks.count_documents(
            {"embedding": {"$exists": True, "$type": "array", "$ne": []}}
        )
    except Exception as exc:  # noqa: BLE001
        status["mongo_error"] = str(exc)
    status["ready"] = bool(status["mongo"])
    return status


@router.post("/chat", response_model=ChatResponse)
async def naive_chat(
    request: NaiveChatRequest,
    orchestrator: NaiveRAGOrchestrator = Depends(get_naive_orchestrator),
):
    """Run the naive-RAG baseline: embed → dense top-K → generate."""
    try:
        return await orchestrator.chat(
            query=request.query,
            user_id=request.user_id,
            org_id=request.org_id,
            session_id=request.session_id,
            patient_id=request.patient_id,
            constraints=request.constraints,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Naive chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Naive RAG failed: {exc}") from exc


@router.post("/compare", response_model=CompareResponse)
async def compare_chat(
    request: NaiveChatRequest,
    naive: NaiveRAGOrchestrator = Depends(get_naive_orchestrator),
    medswin: MedSwinOrchestrator = Depends(get_orchestrator),
):
    """Run naive-RAG and full MedSwin on the same query and return a retrieval diff."""
    try:
        naive_started = time.perf_counter()
        naive_response = await naive.chat(
            query=request.query,
            user_id=request.user_id,
            org_id=request.org_id,
            session_id=request.session_id,
            patient_id=request.patient_id,
            constraints=request.constraints,
            top_k=request.top_k,
        )
        naive_ms = (time.perf_counter() - naive_started) * 1000.0

        medswin_started = time.perf_counter()
        medswin_response = await medswin.chat(
            query=request.query,
            user_id=request.user_id,
            org_id=request.org_id,
            session_id=request.session_id,
            patient_id=request.patient_id,
            constraints=request.constraints,
        )
        medswin_ms = (time.perf_counter() - medswin_started) * 1000.0
        return CompareResponse(
            query=request.query,
            naive=naive_response,
            medswin=medswin_response,
            diff=compare_responses(
                naive_response,
                medswin_response,
                naive_ms=naive_ms,
                medswin_ms=medswin_ms,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline compare failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline compare failed: {exc}") from exc
