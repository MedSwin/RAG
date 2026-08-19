from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.schemas.enums import SourceType
from app.services.adapters.embedding import EmbeddingClient
from app.services.storage import StorageService

logger = logging.getLogger(__name__)
router = APIRouter()


class StoreChunksRequest(BaseModel):
    """Request model for normalizing, embedding, and storing tenant chunks."""
    org_id: str = Field(min_length=1)
    chunks: List[Dict[str, Any]] = Field(min_length=1)
    collection_name: str = "chunks"
    batch_size: int = Field(default=100, ge=1, le=1000)
    source_type: SourceType = SourceType.LIT


class StoreChunksResponse(BaseModel):
    success_count: int
    skipped_count: int = 0
    failed_count: int
    total_chunks: int
    collection_name: str
    message: str


class BuildIndexRequest(BaseModel):
    """Request model for rebuilding the shared ordinary HNSW index."""
    index_path: Optional[str] = None
    mapping_path: Optional[str] = None
    force_rebuild: bool = False


class BuildIndexResponse(BaseModel):
    success: bool
    index_path: str
    mapping_path: str
    total_vectors: int
    message: str


class RefreshEmbeddingsRequest(BaseModel):
    batch_size: Optional[int] = Field(None, ge=1, le=1000)
    org_id: str = Field(min_length=1)


class BenchmarkResetRequest(BaseModel):
    org_id: str
    remove_indexes: bool = True


class StorageStats(BaseModel):
    total_chunks: int
    total_embeddings: int
    source_counts: Dict[str, int] = Field(default_factory=dict)
    active_embeddings: int = 0
    stale_embeddings: int = 0
    cloud_mode: bool = False
    active_embedding_model: Optional[str] = None
    active_embedding_space: Optional[str] = None
    active_embedding_dim: Optional[int] = None
    embedding_refresh: Dict[str, Any] = Field(default_factory=dict)
    index_exists: bool
    index_manifest_path: Optional[str] = None
    index_manifest: Optional[Dict[str, Any]] = None
    index_provenance_valid: bool = False
    index_provenance_error: Optional[str] = None
    index_size: Optional[int] = None
    active_doc_ids: List[str] = Field(default_factory=list)
    last_updated: Optional[datetime] = None


_storage_service = StorageService()


def get_storage_service() -> StorageService:
    return _storage_service


def cleanup_storage_service() -> None:
    _storage_service.cleanup()


def _normalize_chunk(raw: Dict[str, Any], org_id: str, source_type: SourceType) -> Dict[str, Any]:
    """Translate legacy preprocessing records into the production Chunk schema."""
    chunk = dict(raw)
    metadata = dict(chunk.get("metadata") or {})
    embedded_org = str(chunk.get("org_id") or org_id).strip()
    if embedded_org != org_id:
        raise HTTPException(status_code=400, detail="All chunks must match request org_id")

    chunk_id = str(chunk.get("chunk_id") or metadata.get("chunk_id") or "").strip()
    text = str(chunk.get("text") or chunk.get("content") or "").strip()
    doc_id = str(chunk.get("doc_id") or metadata.get("parent_id") or chunk_id).strip()
    if not chunk_id:
        raise HTTPException(status_code=400, detail="Each chunk requires chunk_id or metadata.chunk_id")
    if not text:
        raise HTTPException(status_code=400, detail=f"Chunk {chunk_id} has no text/content")
    if not doc_id:
        raise HTTPException(status_code=400, detail=f"Chunk {chunk_id} has no document identity")

    raw_source = chunk.get("source_type") or source_type.value
    try:
        normalized_source = SourceType(str(raw_source).upper())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Chunk {chunk_id} has invalid source_type {raw_source!r}") from exc

    reliability_defaults = {
        SourceType.CPG: settings.SOURCE_CPG_SCORE,
        SourceType.EMR: settings.SOURCE_EMR_SCORE,
        SourceType.LIT: settings.SOURCE_LIT_SCORE,
        SourceType.SAFETY: settings.EBM_SAFETY_WEIGHT,
    }
    reliability = float(chunk.get("source_reliability", reliability_defaults[normalized_source]))
    grade = chunk.get("evidence_grade")
    if not isinstance(grade, dict):
        grade = {
            "label": normalized_source.value.lower(),
            "score": reliability,
            "source_reliability": reliability,
        }

    return {
        **chunk,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "org_id": org_id,
        "source_type": normalized_source.value,
        "text": text,
        "content": text,  # legacy clients still inspect this field
        "patient_id": chunk.get("patient_id") or metadata.get("patient_id"),
        "section": chunk.get("section") or metadata.get("section"),
        "source_reliability": reliability,
        "evidence_grade": grade,
        "tokenized_text": chunk.get("tokenized_text") or text.lower().split(),
        "metadata": metadata,
        "created_at": chunk.get("created_at") or datetime.now(timezone.utc),
    }


async def _attach_missing_embeddings(chunks: List[Dict[str, Any]], batch_size: int) -> None:
    """Make preprocessing→storage immediately retrievable instead of storing stale chunks."""
    expected_dim = settings.active_embedding_dimension()
    missing = [
        chunk
        for chunk in chunks
        if not isinstance(chunk.get("embedding"), list)
        or len(chunk.get("embedding") or []) != expected_dim
        or chunk.get("embedding_model") != settings.active_embedding_model()
        or chunk.get("embedding_space") != settings.active_embedding_space()
    ]
    if not missing:
        return

    client = EmbeddingClient(settings.active_embedding_url())
    try:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = await client.embed(
                [str(chunk["text"]) for chunk in batch],
                input_type="document" if settings.CLOUD_MODE else None,
            )
            if len(vectors) != len(batch):
                raise RuntimeError(f"Embedding service returned {len(vectors)} vectors for {len(batch)} chunks")
            for chunk, vector in zip(batch, vectors):
                chunk["embedding"] = vector.tolist()
                chunk["embedding_model"] = settings.active_embedding_model()
                chunk["embedding_dim"] = int(len(vector))
                chunk["embedding_space"] = settings.active_embedding_space()
                chunk["embedding_updated_at"] = datetime.now(timezone.utc)
    finally:
        await client.close()


@router.post("/chunks", response_model=StoreChunksResponse)
async def store_chunks(
    request: StoreChunksRequest,
    background_tasks: BackgroundTasks,
    storage_service: StorageService = Depends(get_storage_service),
):
    """Normalize, document-embed, persist, then rebuild the shared ANN."""
    try:
        normalized = [
            _normalize_chunk(raw, request.org_id, request.source_type)
            for raw in request.chunks
        ]
        await _attach_missing_embeddings(normalized, request.batch_size)
        result = await storage_service.store_chunks(
            chunks=normalized,
            collection_name=request.collection_name,
            batch_size=request.batch_size,
        )
        if result["success_count"] > 0:
            # The online retriever uses one configured HNSW artifact. Keep that
            # artifact global across tenants and enforce org isolation when
            # resolving ANN labels through Mongo filters.
            background_tasks.add_task(
                storage_service.build_hnsw_index_async,
                force_rebuild=True,
                org_id=None,
            )
        return StoreChunksResponse(
            success_count=result["success_count"],
            skipped_count=result.get("skipped_count", 0),
            failed_count=result["failed_count"],
            total_chunks=len(normalized),
            collection_name=request.collection_name,
            message=(
                f"Stored {result['success_count']} new chunks, skipped "
                f"{result.get('skipped_count', 0)} existing chunks, and failed {result['failed_count']}"
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error storing chunks: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store chunks: {exc}")


@router.post("/index/build", response_model=BuildIndexResponse)
async def build_index(
    request: BuildIndexRequest,
    storage_service: StorageService = Depends(get_storage_service),
):
    """Rebuild the one shared ordinary ANN over all active tenant vectors."""
    try:
        result = await storage_service.build_hnsw_index_async(
            index_path=request.index_path or settings.HNSW_INDEX_PATH,
            mapping_path=request.mapping_path or settings.HNSW_MAPPING_PATH,
            force_rebuild=request.force_rebuild,
            org_id=None,
        )
        return BuildIndexResponse(
            success=result["success"],
            index_path=result["index_path"],
            mapping_path=result["mapping_path"],
            total_vectors=result["total_vectors"],
            message=result["message"],
        )
    except Exception as exc:
        logger.error("Error building index: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}")


@router.post("/embeddings/refresh")
async def refresh_cloud_embeddings(
    request: RefreshEmbeddingsRequest,
    background_tasks: BackgroundTasks,
    storage_service: StorageService = Depends(get_storage_service),
):
    """Refresh one tenant's vectors, then rebuild the global online ANN."""
    try:
        result = await storage_service.refresh_cloud_embeddings(
            batch_size=request.batch_size,
            org_id=request.org_id,
            rebuild_index=False,
        )
        if result.get("ready"):
            background_tasks.add_task(
                storage_service.build_hnsw_index_async,
                force_rebuild=True,
                org_id=None,
            )
        return result
    except Exception as exc:
        logger.error("Error refreshing cloud embeddings: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to refresh cloud embeddings: {exc}")


@router.post("/benchmark/reset")
async def reset_benchmark_org(
    request: BenchmarkResetRequest,
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        result = await storage_service.clear_benchmark_org(
            org_id=request.org_id,
            remove_indexes=request.remove_indexes,
        )
        return {"success": True, **result}
    except Exception as exc:
        logger.error("Error resetting benchmark org: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset benchmark org: {exc}")


@router.get("/stats", response_model=StorageStats)
async def get_storage_stats(
    org_id: str,
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        return StorageStats(**(await storage_service.get_storage_stats(org_id=org_id)))
    except Exception as exc:
        logger.error("Error getting storage stats: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {exc}")


@router.delete("/chunks")
async def clear_chunks(
    org_id: str,
    collection_name: str = "chunks",
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        result = await storage_service.clear_chunks(collection_name, org_id=org_id)
        return {
            "success": True,
            "org_id": org_id,
            "message": f"Cleared {result['deleted_count']} chunks from {collection_name}",
            "deleted_count": result["deleted_count"],
        }
    except Exception as exc:
        logger.error("Error clearing chunks: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear chunks: {exc}")


@router.get("/chunks/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    org_id: str,
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        chunk = await storage_service.get_chunk(chunk_id, org_id=org_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting chunk: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {exc}")


@router.get("/chunks")
async def list_chunks(
    org_id: str,
    skip: int = 0,
    limit: int = 100,
    source: Optional[str] = None,
    task: Optional[str] = None,
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        filters: Dict[str, Any] = {"org_id": org_id}
        if source:
            filters["metadata.source"] = source
        if task:
            filters["metadata.task"] = task
        chunks = await storage_service.list_chunks(skip=skip, limit=limit, filters=filters)
        return {"chunks": chunks, "total": len(chunks), "skip": skip, "limit": limit, "org_id": org_id}
    except Exception as exc:
        logger.error("Error listing chunks: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list chunks: {exc}")


@router.post("/validate")
async def validate_storage(
    org_id: str,
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        validation = await storage_service.validate_storage(org_id=org_id)
        return {
            "valid": validation["valid"],
            "issues": validation["issues"],
            "org_id": org_id,
            "chunk_count": validation["chunk_count"],
            "index_exists": validation["index_exists"],
            "index_valid": validation["index_valid"],
        }
    except Exception as exc:
        logger.error("Error validating storage: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to validate storage: {exc}")