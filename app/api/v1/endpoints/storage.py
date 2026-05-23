from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timezone

from app.core.state import get_model_manager
from app.core.config import settings
from app.core.database import get_sync_database, get_database
from app.services.storage import StorageService

logger = logging.getLogger(__name__)
router = APIRouter()

class StoreChunksRequest(BaseModel):
    """Request model for storing chunks."""
    chunks: List[Dict[str, Any]]
    collection_name: str = "chunks"
    batch_size: int = 100

class StoreChunksResponse(BaseModel):
    """Response model for storing chunks."""
    success_count: int
    failed_count: int
    total_chunks: int
    collection_name: str
    message: str

class BuildIndexRequest(BaseModel):
    """Request model for building HNSW index."""
    index_path: Optional[str] = None
    mapping_path: Optional[str] = None
    force_rebuild: bool = False

class BuildIndexResponse(BaseModel):
    """Response model for building HNSW index."""
    success: bool
    index_path: str
    mapping_path: str
    total_vectors: int
    message: str

class RefreshEmbeddingsRequest(BaseModel):
    """Request model for active cloud embedding refresh."""
    batch_size: Optional[int] = None
    org_id: Optional[str] = None

class BenchmarkResetRequest(BaseModel):
    """Request model for resetting benchmark-org data."""
    org_id: str
    remove_indexes: bool = True

class StorageStats(BaseModel):
    """Model for storage statistics."""
    total_chunks: int
    total_embeddings: int
    source_counts: Dict[str, int] = Field(default_factory=dict)
    active_embeddings: int = 0
    stale_embeddings: int = 0
    cloud_mode: bool = False
    active_embedding_model: Optional[str] = None
    active_embedding_space: Optional[str] = None
    embedding_refresh: Dict[str, Any] = Field(default_factory=dict)
    index_exists: bool
    index_size: Optional[int] = None
    last_updated: Optional[datetime] = None

def get_storage_service():
    """Dependency to get storage service."""
    try:
        return StorageService()
    except Exception as e:
        logger.error(f"Failed to get storage service: {e}")
        raise HTTPException(status_code=503, detail="Storage service not available")

@router.post("/chunks", response_model=StoreChunksResponse)
async def store_chunks(
    request: StoreChunksRequest,
    background_tasks: BackgroundTasks,
    storage_service = Depends(get_storage_service)
):
    """Store chunks in MongoDB."""
    try:
        # Validate chunks
        if not request.chunks:
            raise HTTPException(status_code=400, detail="No chunks provided")
        
        # Store chunks
        result = await storage_service.store_chunks(
            chunks=request.chunks,
            collection_name=request.collection_name,
            batch_size=request.batch_size
        )
        
        # Trigger index rebuild in background if needed
        if result["success_count"] > 0:
            background_tasks.add_task(
                storage_service.build_hnsw_index_async,
                force_rebuild=True
            )
        
        return StoreChunksResponse(
            success_count=result["success_count"],
            failed_count=result["failed_count"],
            total_chunks=len(request.chunks),
            collection_name=request.collection_name,
            message=f"Successfully stored {result['success_count']} out of {len(request.chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store chunks: {str(e)}")

@router.post("/index/build", response_model=BuildIndexResponse)
async def build_index(
    request: BuildIndexRequest,
    storage_service = Depends(get_storage_service)
):
    """Build HNSW index from stored chunks."""
    try:
        index_path = request.index_path or settings.HNSW_INDEX_PATH
        mapping_path = request.mapping_path or settings.HNSW_MAPPING_PATH
        
        result = await storage_service.build_hnsw_index_async(
            index_path=index_path,
            mapping_path=mapping_path,
            force_rebuild=request.force_rebuild
        )
        
        return BuildIndexResponse(
            success=result["success"],
            index_path=result["index_path"],
            mapping_path=result["mapping_path"],
            total_vectors=result["total_vectors"],
            message=result["message"]
        )
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")

@router.post("/embeddings/refresh")
async def refresh_cloud_embeddings(
    request: RefreshEmbeddingsRequest,
    storage_service = Depends(get_storage_service)
):
    """Refresh stale active-cloud embeddings and rebuild the active index."""
    try:
        return await storage_service.refresh_cloud_embeddings(batch_size=request.batch_size, org_id=request.org_id)
    except Exception as e:
        logger.error(f"Error refreshing cloud embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cloud embeddings: {str(e)}")

@router.post("/benchmark/reset")
async def reset_benchmark_org(
    request: BenchmarkResetRequest,
    storage_service = Depends(get_storage_service)
):
    """Clear benchmark-org data and stale index artifacts for a fresh eval run."""
    try:
        result = await storage_service.clear_benchmark_org(
            org_id=request.org_id,
            remove_indexes=request.remove_indexes,
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Error resetting benchmark org: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset benchmark org: {str(e)}")

@router.get("/stats", response_model=StorageStats)
async def get_storage_stats(storage_service = Depends(get_storage_service)):
    """Get storage statistics."""
    try:
        stats = await storage_service.get_storage_stats()
        return StorageStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

@router.delete("/chunks")
async def clear_chunks(
    collection_name: str = "chunks",
    storage_service = Depends(get_storage_service)
):
    """Clear all chunks from storage."""
    try:
        result = await storage_service.clear_chunks(collection_name)
        
        return {
            "success": True,
            "message": f"Cleared {result['deleted_count']} chunks from {collection_name}",
            "deleted_count": result["deleted_count"]
        }
        
    except Exception as e:
        logger.error(f"Error clearing chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chunks: {str(e)}")

@router.get("/chunks/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    storage_service = Depends(get_storage_service)
):
    """Get a specific chunk by ID."""
    try:
        chunk = await storage_service.get_chunk(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return chunk
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")

@router.get("/chunks")
async def list_chunks(
    skip: int = 0,
    limit: int = 100,
    source: Optional[str] = None,
    task: Optional[str] = None,
    storage_service = Depends(get_storage_service)
):
    """List chunks with optional filtering."""
    try:
        filters = {}
        if source:
            filters["metadata.source"] = source
        if task:
            filters["metadata.task"] = task
        
        chunks = await storage_service.list_chunks(
            skip=skip,
            limit=limit,
            filters=filters
        )
        
        return {
            "chunks": chunks,
            "total": len(chunks),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list chunks: {str(e)}")

@router.post("/validate")
async def validate_storage(
    storage_service = Depends(get_storage_service)
):
    """Validate storage integrity."""
    try:
        validation_result = await storage_service.validate_storage()
        
        return {
            "valid": validation_result["valid"],
            "issues": validation_result["issues"],
            "chunk_count": validation_result["chunk_count"],
            "index_exists": validation_result["index_exists"],
            "index_valid": validation_result["index_valid"]
        }
        
    except Exception as e:
        logger.error(f"Error validating storage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate storage: {str(e)}")
