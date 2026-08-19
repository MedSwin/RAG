"""Tenant-scoped retrieval API backed by the production MedSwin adapters."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.indexing.hnsw import HNSWIndexBuilder
from app.retrieval.dense import DenseRetriever
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    patient_id: Optional[str] = None
    top_k: Optional[int] = None
    use_reranking: bool = True
    initial_top_k: Optional[int] = None
    final_top_k: Optional[int] = None
    source_type: Optional[SourceType] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    rerank_score: Optional[float] = None
    question: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None


class RetrievalResponse(BaseModel):
    query: str
    org_id: str
    documents: List[DocumentResponse]
    total_documents: int
    retrieval_time: float
    used_reranking: bool


class IndexInfo(BaseModel):
    index_path: str
    mapping_path: str
    dimension: int
    total_vectors: int
    index_type: str
    mapping_backend: str


def _to_response(passage: CandidatePassage) -> DocumentResponse:
    qca = _extract_qca_from_text(passage.text, passage.metadata or {})
    # DenseRetriever exposes cosine-like relevance, while the historical API
    # called the field distance. Preserve the response field but return a
    # distance-like value so old clients still sort ascending if they need to.
    dense_score = float(passage.dense_score or 0.0)
    distance = max(0.0, 1.0 - dense_score)
    return DocumentResponse(
        chunk_id=passage.chunk_id,
        doc_id=passage.doc_id,
        source_type=passage.source_type.value,
        content=passage.text,
        metadata=passage.metadata or {},
        distance=distance,
        rerank_score=passage.rerank_score,
        question=qca.get("question"),
        context=qca.get("context"),
        answer=qca.get("answer"),
    )


async def _search(request: QueryRequest) -> RetrievalResponse:
    started = time.perf_counter()
    top_k = max(1, min(int(request.top_k or settings.DEFAULT_TOP_K), settings.MAX_TOP_K))
    initial_top_k = max(
        top_k,
        min(int(request.initial_top_k or settings.RERANK_TOP_K), settings.MAX_TOP_K),
    )
    final_top_k = max(1, min(int(request.final_top_k or settings.FINAL_TOP_K), top_k))

    embedding_client = EmbeddingClient(settings.active_embedding_url())
    reranker_client = RerankerClient(settings.active_reranker_url()) if request.use_reranking else None
    try:
        vectors = await embedding_client.embed(
            [request.query],
            input_type="query" if settings.CLOUD_MODE else None,
        )
        if not vectors:
            raise RuntimeError("Embedding service returned no query vector")

        dense = DenseRetriever()
        candidates = await dense.retrieve(
            vectors[0],
            request.org_id,
            initial_top_k if request.use_reranking else top_k,
            request.source_type,
            request.patient_id,
            request.constraints,
        )

        used_reranking = False
        if request.use_reranking and reranker_client and candidates:
            results = await reranker_client.rerank(
                request.query,
                [candidate.text for candidate in candidates],
                return_logits=True,
            )
            scored: List[CandidatePassage] = []
            for result in results:
                index = int(result.get("index", -1))
                if index < 0 or index >= len(candidates):
                    continue
                candidate = candidates[index]
                score = float(result.get("p_hat", result.get("relevance_score", 0.0)) or 0.0)
                candidate.rerank_score = score
                candidate.calibrated_score = score
                scored.append(candidate)
            if scored:
                scored.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
                candidates = scored[:final_top_k]
                used_reranking = True
            else:
                # Do not silently claim reranking if the provider returned no
                # usable results. A caller requested this stage explicitly.
                raise RuntimeError("Reranker returned no usable ranked candidates")
        else:
            candidates = candidates[:top_k]

        documents = [_to_response(candidate) for candidate in candidates]
        return RetrievalResponse(
            query=request.query,
            org_id=request.org_id,
            documents=documents,
            total_documents=len(documents),
            retrieval_time=time.perf_counter() - started,
            used_reranking=used_reranking,
        )
    finally:
        await embedding_client.close()
        if reranker_client is not None:
            await reranker_client.close()


@router.post("/search", response_model=RetrievalResponse)
async def search_documents(request: QueryRequest):
    """Search one organization's active ANN corpus and optionally rerank it."""
    try:
        return await _search(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Retrieval search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.get("/search", response_model=RetrievalResponse)
async def search_documents_get(
    query: str = Query(..., min_length=1),
    org_id: str = Query(..., min_length=1),
    patient_id: Optional[str] = Query(None),
    top_k: int = Query(settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K),
    use_reranking: bool = Query(True),
    initial_top_k: int = Query(settings.RERANK_TOP_K, ge=1, le=settings.MAX_TOP_K),
    final_top_k: int = Query(settings.FINAL_TOP_K, ge=1, le=settings.MAX_TOP_K),
    source_type: Optional[SourceType] = Query(None),
):
    return await search_documents(
        QueryRequest(
            query=query,
            org_id=org_id,
            patient_id=patient_id,
            top_k=top_k,
            use_reranking=use_reranking,
            initial_top_k=initial_top_k,
            final_top_k=final_top_k,
            source_type=source_type,
        )
    )


@router.get("/index/info", response_model=IndexInfo)
async def get_index_info():
    """Read the active HNSW artifact using JSON or SQLite label mappings."""
    index_path = Path(settings.HNSW_INDEX_PATH)
    mapping_path = Path(settings.HNSW_MAPPING_PATH)
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="HNSW index not found")
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail="HNSW mapping not found")

    builder = HNSWIndexBuilder(settings.active_embedding_dimension())
    try:
        if not builder.load(str(index_path), str(mapping_path)):
            raise RuntimeError("Failed to load HNSW index")
        info = builder.get_index_info()
        return IndexInfo(
            index_path=str(index_path),
            mapping_path=str(mapping_path),
            dimension=int(info["dimension"]),
            total_vectors=int(info["total_vectors"]),
            index_type="HNSW",
            mapping_backend=str(info.get("mapping_backend") or "unknown"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read index info: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get index info: {exc}")
    finally:
        if hasattr(builder.mapping, "close"):
            builder.mapping.close()


def _extract_qca_from_text(content: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """Preserve legacy Q/C/A convenience fields without changing chunk schema."""
    if all(key in metadata for key in ("question", "context", "answer")):
        return {
            "question": str(metadata["question"]),
            "context": str(metadata["context"]),
            "answer": str(metadata["answer"]),
        }

    try:
        import re

        question_pattern = r"(?:Question|Patient Question|Input):\s*(.*?)(?=(?:Context|Answer|Output|Doctor Response):|$)"
        context_pattern = r"(?:Context|Patient Question \(continued\)):\s*(.*?)(?=(?:Answer|Output|Doctor Response):|$)"
        answer_pattern = r"(?:Answer|Output|Doctor Response|Doctor Response \(continued\)):\s*(.*?)$"
        question = re.search(question_pattern, content or "", re.DOTALL | re.IGNORECASE)
        context = re.search(context_pattern, content or "", re.DOTALL | re.IGNORECASE)
        answer = re.search(answer_pattern, content or "", re.DOTALL | re.IGNORECASE)
        return {
            "question": question.group(1).strip() if question else "",
            "context": context.group(1).strip() if context else "",
            "answer": answer.group(1).strip() if answer else "",
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("QCA extraction failed: %s", exc)
        return {"question": "", "context": "", "answer": ""}
