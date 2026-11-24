"""Version-2 unified pipeline API: naive, two-stage, and full."""

from __future__ import annotations

import logging
import os
import time
from hashlib import sha256
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.medswin.naive import NaiveRAGOrchestrator
from app.medswin.orchestrator import MedSwinOrchestrator
from app.repositories.sessions import SessionRepository
from app.repositories.traces import TraceRepository
from app.schemas import ChatResponse, PipelineId
from app.services.adapters.embedding import EmbeddingClient
from app.services.medswin.governance import redacted_trace_summary

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatConstraints(BaseModel):
    """Allowed client constraints. Ablation switches are intentionally absent."""

    model_config = ConfigDict(extra="forbid")
    clinical_scope: Optional[str] = None
    required_facets: Optional[List[Dict[str, Any]]] = None
    source_policy: Optional[str] = None
    min_evidence_grade: Optional[float] = None
    timeframe: Optional[Any] = None
    specialties: Optional[List[str]] = None
    ranking_depth: int = Field(20, ge=1, le=20)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class BenchmarkContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    case_id: str
    question_type: Optional[str] = None
    response_schema: str = "bioasq"


class PipelineChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    user_id: str
    org_id: str
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    constraints: Optional[ChatConstraints] = None
    benchmark: Optional[BenchmarkContext] = None

    def constraint_dict(self) -> Dict[str, Any]:
        return self.constraints.model_dump(exclude_none=True) if self.constraints else {}


class CompareRequest(PipelineChatRequest):
    pipelines: List[PipelineId] = Field(default_factory=lambda: [PipelineId.NAIVE, PipelineId.TWO_STAGE, PipelineId.FULL])


class CompareResponse(BaseModel):
    query: str
    results: Dict[str, ChatResponse]
    adjacent_diffs: Dict[str, Dict[str, Any]]


_naive: Optional[NaiveRAGOrchestrator] = None
_two_stage: Optional[MedSwinOrchestrator] = None
_full: Optional[MedSwinOrchestrator] = None


def _assert_benchmark_allowed(request: PipelineChatRequest) -> None:
    if request.benchmark and not settings.SYSTEM_EVAL_MODE:
        raise HTTPException(status_code=403, detail="Benchmark request metadata requires SYSTEM_EVAL_MODE=true")


def get_naive() -> NaiveRAGOrchestrator:
    global _naive
    if _naive is None:
        _naive = NaiveRAGOrchestrator(embedding_client=EmbeddingClient(settings.active_embedding_url()))
    return _naive


def get_two_stage() -> MedSwinOrchestrator:
    global _two_stage
    if _two_stage is None:
        _two_stage = MedSwinOrchestrator(pipeline=PipelineId.TWO_STAGE)
    return _two_stage


def get_full() -> MedSwinOrchestrator:
    global _full
    if _full is None:
        _full = MedSwinOrchestrator(pipeline=PipelineId.FULL)
    return _full


def _orchestrator(pipeline: PipelineId):
    if pipeline == PipelineId.NAIVE:
        return get_naive()
    if pipeline == PipelineId.TWO_STAGE:
        return get_two_stage()
    return get_full()


async def _chat(pipeline: PipelineId, request: PipelineChatRequest) -> ChatResponse:
    _assert_benchmark_allowed(request)
    instance = _orchestrator(pipeline)
    kwargs = dict(
        query=request.query,
        user_id=request.user_id,
        org_id=request.org_id,
        session_id=request.session_id,
        patient_id=request.patient_id,
        constraints=request.constraint_dict(),
        benchmark=request.benchmark.model_dump() if request.benchmark else None,
    )
    if pipeline == PipelineId.NAIVE:
        kwargs["top_k"] = request.constraints.top_k if request.constraints else None
    response = await instance.chat(**kwargs)
    return response


@router.post("/{pipeline}/chat", response_model=ChatResponse)
async def pipeline_chat(pipeline: PipelineId, request: PipelineChatRequest):
    try:
        return await _chat(pipeline, request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("%s pipeline failed: %s", pipeline.value, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"{pipeline.value} pipeline failed: {exc}") from exc


def _diff(left: ChatResponse, right: ChatResponse) -> Dict[str, Any]:
    left_docs = [row.doc_id for row in left.ranked_evidence]
    right_docs = [row.doc_id for row in right.ranked_evidence]
    return {
        "left_pipeline": left.pipeline.value,
        "right_pipeline": right.pipeline.value,
        "ranking_overlap": sorted(set(left_docs) & set(right_docs)),
        "left_only": sorted(set(left_docs) - set(right_docs)),
        "right_only": sorted(set(right_docs) - set(left_docs)),
        "left_abstained": bool(left.answer_artifact and left.answer_artifact.abstained),
        "right_abstained": bool(right.answer_artifact and right.answer_artifact.abstained),
        "latency_delta_ms": (right.timing_ms.get("total", 0.0) - left.timing_ms.get("total", 0.0)),
    }


@router.post("/compare", response_model=CompareResponse)
async def pipeline_compare(request: CompareRequest):
    requested = list(dict.fromkeys(request.pipelines))
    if not requested:
        raise HTTPException(status_code=400, detail="At least one pipeline is required")
    # Stable rotation makes long-lived compare clients fairer without changing
    # response ordering. The request still returns canonical pipeline keys.
    ordered = sorted(
        requested,
        key=lambda value: (sha256(f"{request.query}\0{value.value}".encode("utf-8")).hexdigest(), value.value),
    )
    results: Dict[str, ChatResponse] = {}
    for pipeline in ordered:
        started = time.perf_counter()
        response = await _chat(pipeline, request)
        response.timing_ms.setdefault("compare_wall", (time.perf_counter() - started) * 1000.0)
        results[pipeline.value] = response
    adjacent: Dict[str, Dict[str, Any]] = {}
    if PipelineId.NAIVE.value in results and PipelineId.TWO_STAGE.value in results:
        adjacent["naive_to_two_stage"] = _diff(results[PipelineId.NAIVE.value], results[PipelineId.TWO_STAGE.value])
    if PipelineId.TWO_STAGE.value in results and PipelineId.FULL.value in results:
        adjacent["two_stage_to_full"] = _diff(results[PipelineId.TWO_STAGE.value], results[PipelineId.FULL.value])
    return CompareResponse(query=request.query, results=results, adjacent_diffs=adjacent)


@router.get("/ready")
async def pipelines_ready():
    return {
        "contract_version": "2.0",
        "pipelines": [item.value for item in PipelineId],
        "cloud_mode": settings.CLOUD_MODE,
        "embedding_model": settings.active_embedding_model(),
        "reranker_model": settings.active_reranker_model() if settings.CLOUD_MODE else None,
        "generation_backend": settings.resolved_generation_backend(),
        "generation_model": settings.active_generation_model(),
        "system_eval_mode": settings.SYSTEM_EVAL_MODE,
    }


@router.get("/traces/{trace_id}")
async def pipeline_trace(trace_id: str, org_id: str, include_details: bool = False):
    trace = await TraceRepository().get_by_id(trace_id, org_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return redacted_trace_summary(trace, include_policy_details=include_details)


@router.get("/sessions/{session_id}")
async def pipeline_session(session_id: str, org_id: str):
    session = await SessionRepository().get_by_id(session_id, org_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "user_id": session.get("user_id"),
        "org_id": org_id,
        "created_at": str(session.get("created_at")),
        "last_active": str(session.get("last_active")),
        "metadata": session.get("metadata", {}),
    }


async def cleanup_pipeline_orchestrators() -> None:
    global _naive, _two_stage, _full
    for instance in (_naive, _two_stage, _full):
        if instance is not None:
            close = getattr(instance, "close", None)
            if close:
                await close()
    _naive = _two_stage = _full = None
