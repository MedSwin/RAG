"""MedSwin API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from datetime import timezone
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
    rate_limit_stats: Optional[Dict[str, Any]] = None
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
            rate_limit_stats=summary.get("rate_limit_stats"),
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
        
        # Validate source type
        try:
            source_type_enum = SourceType(source_type.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type}")
        
        doc_repo = DocumentRepository()
        chunk_repo = ChunkRepository()
        
        # Root Cause vs Logic: PMC literature ingest does not require the embedding
        # model to be loaded up front. The previous implementation fetched the
        # tokenizer unconditionally, which turned an optional preprocessing path
        # into a hard 503 for otherwise valid document ingests. We keep the route
        # resilient and rely on section-aware chunking plus persisted metadata.

        # Root Cause vs Logic: embedding each document immediately forced one
        # cloud request per article, so judged-pool ingests degenerated into
        # thousands of single-item calls. We stage the documents first, attach
        # embeddings in bulk, and then persist the fully prepared records.
        staged_documents: list[tuple[Document, list[Chunk]]] = []
        staged_chunks: list[Chunk] = []
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

            staged_documents.append((document, chunks))
            staged_chunks.extend(chunks)

        if staged_chunks and settings.CLOUD_MODE:
            await _attach_active_embeddings(staged_chunks)

        for document, chunks in staged_documents:
            await doc_repo.create(document, org_id)

            # Root Cause vs Logic: Some PMC records are metadata-only or have
            # no recoverable text/section content, which makes the chunk list
            # empty. Mongo rejects insert_many([]), so we only persist chunks
            # when there is at least one materialized passage.
            if chunks:
                await chunk_repo.create_many(chunks, org_id)

            results.append({
                "doc_id": document.doc_id,
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


async def _attach_active_embeddings(chunks: List[Any]) -> None:
    """Attach active cloud embeddings to chunks before persistence.

    Motivation vs Logic: In cloud mode, retrieval depends on the active Azure
    embedding vector space. Persisting chunks without embeddings leaves them
    invisible until an external refresh, so ingest attaches embeddings eagerly
    and the refresh job remains a repair path for stale data.
    """
    client = EmbeddingClient(settings.active_embedding_url())
    try:
        # Motivation vs Logic: the embeddings service accepts batched input,
        # so we preserve throughput by attaching vectors in moderate batches
        # rather than issuing one request per article or one giant payload.
        # Cloud providers are especially sensitive to oversized embedding
        # requests, so we cap the batch size lower in cloud mode to favor
        # reliable corpus builds over peak throughput.
        batch_size = max(1, int(settings.CLOUD_EMBED_BATCH_SIZE if settings.CLOUD_MODE else settings.BATCH_SIZE))
        embeddings = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings = await client.embed([chunk.text for chunk in batch])
            embeddings.extend(batch_embeddings)
            if settings.CLOUD_MODE and settings.CLOUD_EMBED_BATCH_DELAY_S > 0 and start + batch_size < len(chunks):
                await asyncio.sleep(settings.CLOUD_EMBED_BATCH_DELAY_S)
    finally:
        await client.close()
    if len(embeddings) != len(chunks):
        raise RuntimeError(f"Embedding service returned {len(embeddings)} vectors for {len(chunks)} chunks")
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()
        chunk.embedding_model = settings.CLOUD_EMBEDDING
        chunk.embedding_dim = int(len(embedding))
        chunk.embedding_space = settings.active_embedding_space()
        chunk.embedding_updated_at = datetime.now(timezone.utc)


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
