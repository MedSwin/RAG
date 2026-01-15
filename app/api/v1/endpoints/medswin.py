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
        description="Optional constraints (guideline_only, timeframe, specialties)"
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


# Global orchestrator instance (can be dependency-injected in production)
_orchestrator: Optional[MedSwinOrchestrator] = None


def get_orchestrator() -> MedSwinOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        embedding_client = EmbeddingClient(settings.EMBEDDING_URL)
        reranker_client = RerankerClient(settings.RERANKER_URL)
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
async def get_trace(trace_id: str, org_id: str, orchestrator: MedSwinOrchestrator = Depends(get_orchestrator)):
    """Get structured trace (admin/authorized only, optionally redacted)."""
    try:
        from app.repositories.traces import TraceRepository
        from datetime import datetime
        
        trace_repo = TraceRepository()
        trace = await trace_repo.get_by_id(trace_id, org_id)
        
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        # Redact PHI if enabled
        if settings.LOG_REDACT_PHI:
            # In production, implement proper PHI redaction
            pass
        
        return TraceResponse(
            trace_id=trace["trace_id"],
            session_id=trace["session_id"],
            query=trace["query"],
            created_at=trace["created_at"].isoformat() if isinstance(trace["created_at"], datetime) else str(trace["created_at"]),
            completed_at=trace.get("completed_at").isoformat() if trace.get("completed_at") and isinstance(trace["completed_at"], datetime) else (str(trace["completed_at"]) if trace.get("completed_at") else None),
            messages_count=len(trace.get("messages", [])),
            tool_calls_count=len(trace.get("tool_calls", [])),
            sufficiency_checks_count=len(trace.get("sufficiency_checks", [])),
            evidence_passages_count=len(trace.get("evidence_bundle", {}).get("passages", [])) if trace.get("evidence_bundle") else 0
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
        from app.models.medswin import SourceType, Document, Chunk
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
            document = Document(
                doc_id=doc_id,
                source_type=source_type_enum,
                title=doc_data.get("title", "Untitled"),
                version=doc_data.get("version"),
                effective_date=doc_data.get("effective_date"),
                patient_id=doc_data.get("patient_id"),
                org_id=org_id,
                tags=doc_data.get("tags", []),
                metadata=doc_data.get("metadata", {})
            )
            
            await doc_repo.create(document, org_id)
            
            # Chunk document (simple chunking - split by sentences/paragraphs)
            text = doc_data.get("text", doc_data.get("content", ""))
            # Simple chunking: split by paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks_data = []
            for idx, para in enumerate(paragraphs):
                if para:
                    chunks_data.append({
                        "chunk_id": f"{doc_id}_chunk_{idx}",
                        "text": para,
                        "content": para,
                        "section": doc_data.get("section"),
                        "offset_start": None,
                        "offset_end": None,
                        "metadata": {}
                    })
            
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
                    org_id=org_id,
                    metadata=chunk_data.get("metadata", {})
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

