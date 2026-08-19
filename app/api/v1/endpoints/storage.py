from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from app.core.config import settings
from app.services.storage import StorageService

logger = logging.getLogger(__name__)
router = APIRouter()


class StoreChunksRequest(BaseModel):
    """Request model for storing tenant-scoped chunks."""
    org_id: str
    chunks: List[Dict[str, Any]]
    collection_name: str = "chunks"
    batch_size: int = 100


class StoreChunksResponse(BaseModel):
    """Response model for storing chunks."""
    success_count: int
    skipped_count: int = 0
    failed_count: int
    total_chunks: int
    collection_name: str
    message: str


class BuildIndexRequest(BaseModel):
    """Request model for building an ordinary HNSW index."""
    index_path: Optional[str] = None
    mapping_path: Optional[str] = None
    force_rebuild: bool = False
    org_id: str


class BuildIndexResponse(BaseModel):
    success: bool
    index_path: str
    mapping_path: str
    total_vectors: int
    message: str


class RefreshEmbeddingsRequest(BaseModel):
    """Request model for active cloud embedding refresh."""
    batch_size: Optional[int] = None
    org_id: str


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


# One service per API process. StorageService owns a lazily-created thread pool;
# constructing it per request would leak executors under sustained admin usage.
_storage_service = StorageService()


def get_storage_service() -> StorageService:
    return _storage_service


def cleanup_storage_service() -> None:
    _storage_service.cleanup()


@router.post("/chunks", response_model=StoreChunksResponse)
async def store_chunks(
    request: StoreChunksRequest,
    background_tasks: BackgroundTasks,
    storage_service: StorageService = Depends(get_storage_service),
):
    """Store chunks for exactly one organization and refresh that index."""
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")

    normalized: List[Dict[str, Any]] = []
    for raw in request.chunks:
        chunk = dict(raw)
        embedded_org = str(chunk.get("org_id") or request.org_id).strip()
        if embedded_org != request.org_id:
            raise HTTPException(status_code=400, detail="All chunks must match request org_id")
        chunk["org_id"] = request.org_id
        normalized.append(chunk)

    try:
        result = await storage_service.store_chunks(
            chunks=normalized,
            collection_name=request.collection_name,
            batch_size=request.batch_size,
        )
        if result["success_count"] > 0:
            background_tasks.add_task(
                storage_service.build_hnsw_index_async,
                force_rebuild=True,
                org_id=request.org_id,
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
    """Build HNSW index for one organization."""
    try:
        result = await storage_service.build_hnsw_index_async(
            index_path=request.index_path or settings.HNSW_INDEX_PATH,
            mapping_path=request.mapping_path or settings.HNSW_MAPPING_PATH,
            force_rebuild=request.force_rebuild,
            org_id=request.org_id,
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
    storage_service: StorageService = Depends(get_storage_service),
):
    try:
        return await storage_service.refresh_cloud_embeddings(
            batch_size=request.batch_size,
            org_id=request.org_id,
        )
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
    """Clear only the requested organization's chunks."""
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
