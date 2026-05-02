"""MedSwin API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import uuid

from app.services.medswin.orchestrator import MedSwinOrchestrator
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient
from app.core.config import settings
from app.models.medswin import ChatResponse, AuditTrace
from app.services.medswin.governance import redacted_trace_summary

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for MedSwin chat."""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID (creates new if missing)")
    user_id: str = Field(..., description="User ID")
    org_id: str = Field(..., description="Organization ID")
    patient_id: Optional[str] = Field(None, description="Optional patient ID")
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional constraints (clinical_scope, required_facets, source_policy, min_evidence_grade, timeframe, specialties)"
    )


class SessionResponse(BaseModel):
    """Response model for session."""
    session_id: str
    user_id: str
    org_id: str
    created_at: str
    last_active: str
    metadata: Dict[str, Any]


class TraceResponse(BaseModel):
    """Response model for trace (redacted)."""
    trace_id: str
    session_id: str
    query: str
    created_at: str
    completed_at: Optional[str]
    messages_count: int
    tool_calls_count: int
    sufficiency_checks_count: int
    evidence_passages_count: int
    policy_decisions: Optional[List[Dict[str, Any]]] = None
    facet_coverage: Optional[List[Dict[str, Any]]] = None
    contradictions: Optional[List[Dict[str, Any]]] = None


# Global orchestrator instance (can be dependency-injected in production)
_orchestrator: Optional[MedSwinOrchestrator] = None


def get_orchestrator() -> MedSwinOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        embedding_client = EmbeddingClient(settings.active_embedding_url())
        reranker_client = RerankerClient(settings.active_reranker_url())
        _orchestrator = MedSwinOrchestrator(
            embedding_client=embedding_client,
            reranker_client=reranker_client
        )
    return _orchestrator


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, orchestrator: MedSwinOrchestrator = Depends(get_orchestrator)):
    """Process a chat query through MedSwin pipeline.
    
    This endpoint orchestrates the multi-agent conversation system:
    - Supervisor normalizes query
    - Evidence Retriever (Agent 1) searches CPG + EMR indices
    - EMR Summariser (Agent 2) produces structured patient state
    - Guideline Synthesiser (Agent 3) extracts recommendations
    - Safety Critic checks for conflicts and unsafe suggestions
    - Supervisor generates final answer with citations
    """
    try:
        response = await orchestrator.chat(
            query=request.query,
            user_id=request.user_id,
            org_id=request.org_id,
            session_id=request.session_id,
            patient_id=request.patient_id,
            constraints=request.constraints
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, org_id: str, orchestrator: MedSwinOrchestrator = Depends(get_orchestrator)):
    """Get session summary and last N turns (redacted)."""
    try:
        from app.repositories.sessions import SessionRepository
        session_repo = SessionRepository()
        session = await session_repo.get_by_id(session_id, org_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            session_id=session["session_id"],
            user_id=session["user_id"],
            org_id=session["org_id"],
            created_at=session["created_at"].isoformat() if isinstance(session["created_at"], datetime) else str(session["created_at"]),
            last_active=session["last_active"].isoformat() if isinstance(session["last_active"], datetime) else str(session["last_active"]),
            metadata=session.get("metadata", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    org_id: str,
    include_details: bool = False,
    orchestrator: MedSwinOrchestrator = Depends(get_orchestrator)
):
    """Get structured trace (admin/authorized only, optionally redacted)."""
    try:
        from app.repositories.traces import TraceRepository
        from datetime import datetime
        
        trace_repo = TraceRepository()
        trace = await trace_repo.get_by_id(trace_id, org_id)
        
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        include_policy_details = include_details and settings.TRACE_INCLUDE_POLICY_DETAILS
        summary = redacted_trace_summary(trace, include_policy_details=include_policy_details)
        return TraceResponse(
            trace_id=summary["trace_id"],
            session_id=summary["session_id"],
            query=summary["query"],
            created_at=summary["created_at"].isoformat() if isinstance(summary["created_at"], datetime) else str(summary["created_at"]),
            completed_at=summary.get("completed_at").isoformat() if summary.get("completed_at") and isinstance(summary["completed_at"], datetime) else (str(summary["completed_at"]) if summary.get("completed_at") else None),
            messages_count=summary["messages_count"],
            tool_calls_count=summary["tool_calls_count"],
            sufficiency_checks_count=summary["sufficiency_checks_count"],
            evidence_passages_count=summary["evidence_passages_count"],
            policy_decisions=summary.get("policy_decisions"),
            facet_coverage=summary.get("facet_coverage"),
            contradictions=summary.get("contradictions"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get trace failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")


@router.post("/ingest")
async def ingest_documents(
    source_type: str,
    org_id: str,
    documents: List[Dict[str, Any]],
    orchestrator: MedSwinOrchestrator = Depends(get_orchestrator)
):
    """Ingest documents (CPG or EMR).
    
    This endpoint supports ingestion of:
    - CPG documents (pdf/txt/json)
    - EMR notes (structured or unstructured)
    
    Ensures chunk metadata includes source_type, version, patient_id, timestamp.
    """
    try:
        from app.models.medswin import SourceType, Document, Chunk, EvidenceGrade
        from app.repositories.documents import DocumentRepository
        from app.repositories.chunks import ChunkRepository
        from app.services.preprocessing import PreprocessingService
        from app.core.state import get_model_manager
        
        # Validate source type
        try:
            source_type_enum = SourceType(source_type.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")
        
        doc_repo = DocumentRepository()
        chunk_repo = ChunkRepository()
        
        # Get tokenizer for preprocessing
        try:
            tokenizer, _, _, _ = get_model_manager().get_embedding_model()
            preprocessing_service = PreprocessingService(tokenizer)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Preprocessing service not available: {str(e)}")
        
        # Process each document
        results = []
        for doc_data in documents:
            # Create document
            doc_id = doc_data.get("doc_id") or str(uuid.uuid4())
            evidence_grade = _coerce_evidence_grade(doc_data.get("evidence_grade"), source_type_enum)
            document = Document(
                doc_id=doc_id,
                source_type=source_type_enum,
                title=doc_data.get("title", "Untitled"),
                version=doc_data.get("version"),
                effective_date=doc_data.get("effective_date"),
                patient_id=doc_data.get("patient_id"),
                org_id=org_id,
                tags=doc_data.get("tags", []),
                source_reliability=float(doc_data.get("source_reliability", evidence_grade.source_reliability)),
                evidence_grade=evidence_grade,
                metadata={
                    **doc_data.get("metadata", {}),
                    "version": doc_data.get("version"),
                    "effective_date": doc_data.get("effective_date"),
                    "source_reliability": float(doc_data.get("source_reliability", evidence_grade.source_reliability)),
                    "evidence_grade": evidence_grade.model_dump(),
                }
            )
            
            await doc_repo.create(document, org_id)
            
            # Chunk document while preserving existing section/offset metadata when provided.
            text = doc_data.get("text", doc_data.get("content", ""))
            chunks_data = doc_data.get("chunks") or _section_aware_chunks(doc_id, text, doc_data.get("section"))
            
            # Create chunks
            chunks = []
            for chunk_data in chunks_data:
                chunk_id = chunk_data.get("chunk_id") or str(uuid.uuid4())
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_type=source_type_enum,
                    text=chunk_data.get("text", chunk_data.get("content", "")),
                    section=chunk_data.get("section"),
                    offset_start=chunk_data.get("offset_start"),
                    offset_end=chunk_data.get("offset_end"),
                    patient_id=doc_data.get("patient_id"),
                    guideline_version=doc_data.get("version"),
                    timestamp=chunk_data.get("timestamp") or doc_data.get("timestamp"),
                    org_id=org_id,
                    evidence_grade=_coerce_evidence_grade(chunk_data.get("evidence_grade") or doc_data.get("evidence_grade"), source_type_enum),
                    source_reliability=float(chunk_data.get("source_reliability", doc_data.get("source_reliability", evidence_grade.source_reliability))),
                    metadata={
                        **doc_data.get("metadata", {}),
                        **chunk_data.get("metadata", {}),
                        "title": doc_data.get("title", "Untitled"),
                        "version": doc_data.get("version"),
                        "guideline_version": doc_data.get("version"),
                        "effective_date": doc_data.get("effective_date"),
                        "timestamp": chunk_data.get("timestamp") or doc_data.get("timestamp"),
                        "source_reliability": float(chunk_data.get("source_reliability", doc_data.get("source_reliability", evidence_grade.source_reliability))),
                        "evidence_grade": _coerce_evidence_grade(chunk_data.get("evidence_grade") or doc_data.get("evidence_grade"), source_type_enum).model_dump(),
                    },
                    tokenized_text=chunk_data.get("tokenized_text") or chunk_data.get("text", chunk_data.get("content", "")).lower().split()
                )
                chunks.append(chunk)
            
            await chunk_repo.create_many(chunks, org_id)
            
            results.append({
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "status": "success"
            })
        
        return {
            "ingested": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest documents: {str(e)}")


def _coerce_evidence_grade(raw: Any, source_type) -> Any:
    """Coerce request evidence metadata into an EvidenceGrade."""
    from app.models.medswin import EvidenceGrade, SourceType

    if isinstance(raw, EvidenceGrade):
        return raw
    if isinstance(raw, dict):
        return EvidenceGrade(**raw)
    defaults = {
        SourceType.CPG: ("guideline", settings.EBM_CPG_WEIGHT, settings.SOURCE_CPG_SCORE),
        SourceType.EMR: ("emr", settings.EBM_EMR_WEIGHT, settings.SOURCE_EMR_SCORE),
        SourceType.LIT: ("literature", settings.SOURCE_LIT_SCORE, settings.SOURCE_LIT_SCORE),
    }
    label, score, reliability = defaults.get(source_type, ("ungraded", 0.5, 0.5))
    return EvidenceGrade(label=str(raw or label), score=score, source_reliability=reliability)


def _section_aware_chunks(doc_id: str, text: str, default_section: Optional[str]) -> List[Dict[str, Any]]:
    """Split text into chunks while keeping section headings and offsets.

    Motivation vs Logic: Clinical chunks need section provenance for guideline
    recommendations and contraindications. This fallback preserves headings and
    offsets instead of flattening every document into anonymous paragraphs.
    """
    chunks: List[Dict[str, Any]] = []
    if not text:
        return chunks
    offset = 0
    section = default_section
    buffer: List[str] = []
    buffer_start = 0
    chunk_idx = 0

    for paragraph in [part for part in text.split("\n\n") if part.strip()]:
        stripped = paragraph.strip()
        start = text.find(paragraph, offset)
        if start < 0:
            start = offset
        offset = start + len(paragraph)
        is_heading = len(stripped) <= 120 and not stripped.endswith(".") and len(stripped.split()) <= 12
        if is_heading and buffer:
            body = "\n\n".join(buffer).strip()
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "text": body,
                "content": body,
                "section": section,
                "offset_start": buffer_start,
                "offset_end": start,
                "metadata": {},
            })
            chunk_idx += 1
            buffer = []
        if is_heading:
            section = stripped
            buffer_start = offset
            continue
        if not buffer:
            buffer_start = start
        buffer.append(stripped)

    if buffer:
        body = "\n\n".join(buffer).strip()
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
            "text": body,
            "content": body,
            "section": section,
            "offset_start": buffer_start,
            "offset_end": len(text),
            "metadata": {},
        })
    return chunks
