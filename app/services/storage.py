try:
    import pymongo
except ModuleNotFoundError:
    pymongo = None
try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

from app.core.config import settings
from app.core.database import get_sync_database, get_database
from app.services.adapters.embedding import EmbeddingClient
try:
    from app.core.indexing import (
        HNSWIndexBuilder,
        FAISSIndexBuilder,
        TreeIndexBuilder
    )
except ModuleNotFoundError:
    HNSWIndexBuilder = FAISSIndexBuilder = TreeIndexBuilder = None
from app.services.strategy import (
    IndexStrategyManager,
    IndexType,
    IndexStrategy,
    analyze_chunk_characteristics
)

logger = logging.getLogger(__name__)

EMBEDDING_REFRESH_STATUS: Dict[str, Any] = {
    "running": False,
    "ready": not settings.CLOUD_MODE,
    "updated": 0,
    "stale": 0,
    "error": None,
    "embedding_space": settings.active_embedding_space(),
    "started_at": None,
    "completed_at": None,
}


def _index_manifest_path(index_path: str | Path) -> Path:
    """Return the provenance sidecar path for a given index artifact."""
    path = Path(index_path)
    return path.with_name(f"{path.name}.manifest.json")


def _corpus_signature(chunk_ids: List[str], org_id: str, embedding_space: str) -> str:
    """Derive a deterministic corpus signature from the active index contents."""
    digest = hashlib.sha256()
    digest.update(org_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(embedding_space.encode("utf-8"))
    digest.update(b"\0")
    for chunk_id in sorted(chunk_ids):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

class StorageService:
    """Service for managing data storage and indexing."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _write_index_manifest(self, index_path: str, manifest: Dict[str, Any]) -> str:
        manifest_path = _index_manifest_path(index_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        return str(manifest_path)

    def _read_index_manifest(self, index_path: str) -> Dict[str, Any] | None:
        manifest_path = _index_manifest_path(index_path)
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read index manifest %s: %s", manifest_path, exc)
            return None

    def _manifest_matches_active_scope(self, manifest: Dict[str, Any] | None, org_id: Optional[str]) -> bool:
        if not manifest:
            return False
        if org_id and manifest.get("org_id") != org_id:
            return False
        return (
            manifest.get("embedding_space") == settings.active_embedding_space()
            and manifest.get("embedding_model") == settings.active_embedding_model()
            and int(manifest.get("embedding_dim") or 0) == settings.active_embedding_dimension()
        )
    
    async def store_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        collection_name: str = "chunks",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Store chunks in MongoDB asynchronously."""
        try:
            # Run storage in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._store_chunks_sync,
                chunks,
                collection_name,
                batch_size
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def _store_chunks_sync(
        self, 
        chunks: List[Dict[str, Any]], 
        collection_name: str,
        batch_size: int
    ) -> Dict[str, Any]:
        """Synchronous chunk storage function."""
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DATABASE]
        coll = db[collection_name]
        
        success_count = 0
        failed_count = 0
        failed_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Filter out existing chunk_ids
            existing_ids = [
                doc['chunk_id'] for doc in coll.find(
                    {"chunk_id": {"$in": [chunk['chunk_id'] for chunk in batch]}}, 
                    {"chunk_id": 1}
                )
            ]
            new_batch = [chunk for chunk in batch if chunk['chunk_id'] not in existing_ids]
            
            if not new_batch:
                logger.info(f"Batch {i//batch_size + 1} skipped: all chunks already exist")
                continue
            
            # Prepare new batch
            for chunk in new_batch:
                if 'metadata' in chunk and 'created_timestamp' in chunk['metadata']:
                    chunk['metadata']['created_timestamp'] = chunk['metadata']['created_timestamp'].replace(tzinfo=timezone.utc)
            
            try:
                result = coll.insert_many(new_batch, ordered=False)
                success_count += len(result.inserted_ids)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} chunks")
            except pymongo.errors.BulkWriteError as bwe:
                logger.error(f"Batch error: {bwe.details}")
                failed_chunks.extend(new_batch)
                failed_count += len(new_batch)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                failed_chunks.extend(new_batch)
                failed_count += len(new_batch)
        
        # Save failed chunks
        if failed_chunks:
            failed_path = Path(settings.DATA_DIR) / "failed_chunks.json"
            with open(failed_path, 'w') as f:
                json.dump(failed_chunks, f, default=str)
            logger.info(f"{len(failed_chunks)} failed chunks saved to {failed_path}")
        
        client.close()
        
        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "total_chunks": len(chunks)
        }
    
    async def build_hnsw_index_async(
        self,
        index_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        force_rebuild: bool = False,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build HNSW index asynchronously."""
        try:
            index_path = index_path or settings.HNSW_INDEX_PATH
            mapping_path = mapping_path or settings.HNSW_MAPPING_PATH
            
            # Root Cause vs Logic: a cached index file alone does not prove the
            # index belongs to the current benchmark org or embedding space.
            if not force_rebuild and Path(index_path).exists():
                manifest = self._read_index_manifest(index_path)
                if self._manifest_matches_active_scope(manifest, org_id):
                    return {
                        "success": True,
                        "index_path": index_path,
                        "mapping_path": mapping_path,
                        "manifest_path": str(_index_manifest_path(index_path)),
                        "index_manifest": manifest,
                        "total_vectors": int(manifest.get("total_vectors") or 0) if manifest else 0,
                        "message": "Index already exists with matching provenance",
                    }
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "index_manifest": manifest,
                    "total_vectors": 0,
                    "message": "Index exists but provenance does not match the active org or embedding space",
                }
            
            # Run index building in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._build_hnsw_index_sync,
                index_path,
                mapping_path,
                org_id,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error building HNSW index: {e}")
            raise

    async def refresh_cloud_embeddings(self, batch_size: Optional[int] = None, org_id: Optional[str] = None) -> Dict[str, Any]:
        """Refresh stale chunk embeddings for the active cloud embedding space.

        Motivation vs Logic: Cloud embedding models occupy a different vector
        space from local models, so mixed indexes would silently corrupt
        retrieval. Refresh runs in the background and marks readiness only after
        chunks are re-embedded and the active index is rebuilt.
        """
        if not settings.CLOUD_MODE:
            EMBEDDING_REFRESH_STATUS.update({"running": False, "ready": True, "error": None})
            return EMBEDDING_REFRESH_STATUS.copy()

        batch_size = batch_size or settings.BATCH_SIZE
        batch_size = max(1, min(int(batch_size), int(settings.CLOUD_EMBED_BATCH_SIZE)))
        EMBEDDING_REFRESH_STATUS.update({
            "running": True,
            "ready": False,
            "updated": 0,
            "error": None,
            "embedding_space": settings.active_embedding_space(),
            "started_at": datetime.now(timezone.utc),
            "completed_at": None,
        })

        try:
            db = get_database()
            coll = db["chunks"]
            stale_filter = self._stale_embedding_filter(org_id=org_id)
            stale_count = await coll.count_documents(stale_filter)
            EMBEDDING_REFRESH_STATUS["stale"] = stale_count

            client = EmbeddingClient(settings.active_embedding_url())
            try:
                while True:
                    cursor = coll.find(stale_filter).limit(batch_size)
                    chunks = await cursor.to_list(length=batch_size)
                    if not chunks:
                        break
                    texts = [chunk.get("text") or chunk.get("content", "") for chunk in chunks]
                    embeddings = await client.embed(texts)
                    for chunk, embedding in zip(chunks, embeddings):
                        await coll.update_one(
                            {"_id": chunk["_id"]},
                            {"$set": {
                                "embedding": embedding.tolist(),
                                "embedding_model": settings.CLOUD_EMBEDDING,
                                "embedding_dim": int(len(embedding)),
                                "embedding_space": settings.active_embedding_space(),
                                "embedding_updated_at": datetime.now(timezone.utc),
                            }},
                        )
                        EMBEDDING_REFRESH_STATUS["updated"] += 1
                    if batch_size > 0 and settings.CLOUD_EMBED_BATCH_DELAY_S > 0:
                        await asyncio.sleep(settings.CLOUD_EMBED_BATCH_DELAY_S)
            finally:
                await client.close()

            await self.build_hnsw_index_async(force_rebuild=True, org_id=org_id)
            EMBEDDING_REFRESH_STATUS.update({
                "running": False,
                "ready": True,
                "completed_at": datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.error(f"Cloud embedding refresh failed: {e}", exc_info=True)
            EMBEDDING_REFRESH_STATUS.update({
                "running": False,
                "ready": False,
                "error": str(e),
                "completed_at": datetime.now(timezone.utc),
            })
        return EMBEDDING_REFRESH_STATUS.copy()

    def get_embedding_refresh_status(self) -> Dict[str, Any]:
        return EMBEDDING_REFRESH_STATUS.copy()

    def _stale_embedding_filter(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        expected_dim = settings.active_embedding_dimension()
        filter_dict: Dict[str, Any] = {
            "$or": [
                {"embedding": {"$exists": False}},
                {"embedding": []},
                {"embedding_space": {"$ne": settings.active_embedding_space()}},
                {"embedding_space": {"$exists": False}},
                {"embedding_model": {"$ne": settings.CLOUD_EMBEDDING}},
                {"embedding_model": {"$exists": False}},
                {"embedding_dim": {"$ne": expected_dim}},
                {"embedding_dim": {"$exists": False}},
            ]
        }
        if org_id:
            filter_dict["org_id"] = org_id
        return filter_dict
    
    def _build_hnsw_index_sync(
        self,
        index_path: str,
        mapping_path: str,
        org_id: Optional[str] = None,
        index_type: IndexType = IndexType.HNSW
    ) -> Dict[str, Any]:
        """Synchronous index building function using modular index builders."""
        try:
            # Get database
            db = get_sync_database()
            coll = db['chunks']
            
            # Get all chunks with embeddings
            chunks = list(coll.find(
                self._index_embedding_filter(org_id=org_id),
                {"chunk_id": 1, "embedding": 1, "embedding_dim": 1, "metadata": 1}
            ))
            
            if not chunks:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "total_vectors": 0,
                    "message": "No chunks with embeddings found"
                }
            
            # Get embedding dimension
            embedding_dim = chunks[0].get('embedding_dim', settings.active_embedding_dimension())
            
            # Prepare embeddings and mapping
            embeddings = []
            chunk_ids = []
            
            for chunk in chunks:
                if len(chunk.get("embedding") or []) != embedding_dim:
                    logger.warning(
                        "Skipping chunk %s with mismatched embedding dimension %s (expected %s)",
                        chunk.get("chunk_id"),
                        len(chunk.get("embedding") or []),
                        embedding_dim,
                    )
                    continue
                embeddings.append(chunk['embedding'])
                chunk_ids.append(chunk['chunk_id'])
            if not embeddings:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "total_vectors": 0,
                    "message": "No chunks with active-dimension embeddings found"
                }
            
            # Initialize strategy manager
            strategy_manager = IndexStrategyManager()
            
            # Select appropriate index builder based on type
            if index_type == IndexType.HNSW:
                config = strategy_manager.get_index_config(
                    IndexStrategy.HNSW_ONLY,
                    embedding_dim,
                    len(chunks)
                )
                builder = HNSWIndexBuilder(embedding_dim, config.get("hnsw"))
                
            elif index_type == IndexType.FAISS_IVF:
                config = strategy_manager.get_index_config(
                    IndexStrategy.FAISS_ONLY,
                    embedding_dim,
                    len(chunks)
                )
                builder = FAISSIndexBuilder(embedding_dim, config.get("faiss_ivf"))
                
            elif index_type == IndexType.FAISS_TREE:
                config = strategy_manager.get_index_config(
                    IndexStrategy.TREE_ONLY,
                    embedding_dim,
                    len(chunks)
                )
                builder = TreeIndexBuilder(embedding_dim, config.get("faiss_tree"))
                
            else:
                # Default to HNSW
                config = strategy_manager.get_index_config(
                    IndexStrategy.HNSW_ONLY,
                    embedding_dim,
                    len(chunks)
                )
                builder = HNSWIndexBuilder(embedding_dim, config.get("hnsw"))
            
            # Build the index
            result = builder.build(embeddings, chunk_ids, index_path, mapping_path)
            manifest = {
                "org_id": org_id,
                "index_type": index_type.value,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "manifest_path": str(_index_manifest_path(index_path)),
                "embedding_space": settings.active_embedding_space(),
                "embedding_model": settings.active_embedding_model(),
                "embedding_dim": embedding_dim,
                "chunk_count": len(chunks),
                "total_vectors": len(embeddings),
                "source_counts": {
                    "CPG": sum(1 for chunk in chunks if chunk.get("source_type") == "CPG"),
                    "EMR": sum(1 for chunk in chunks if chunk.get("source_type") == "EMR"),
                    "LIT": sum(1 for chunk in chunks if chunk.get("source_type") == "LIT"),
                },
                "corpus_signature": _corpus_signature(
                    chunk_ids,
                    org_id or "all",
                    settings.active_embedding_space(),
                ),
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = self._write_index_manifest(index_path, manifest)
            result["manifest_path"] = manifest_path
            result["index_manifest"] = manifest

            logger.info(
                "%s index built for %s: %s",
                index_type.value.upper(),
                org_id or "all orgs",
                result["message"],
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "manifest_path": str(_index_manifest_path(index_path)),
                "total_vectors": 0,
                "message": f"Index building failed: {str(e)}"
            }

    def _index_embedding_filter(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        filter_dict: Dict[str, Any] = {"embedding": {"$exists": True, "$ne": []}}
        if settings.CLOUD_MODE:
            filter_dict["embedding_space"] = settings.active_embedding_space()
            filter_dict["embedding_model"] = settings.CLOUD_EMBEDDING
            filter_dict["embedding_dim"] = settings.active_embedding_dimension()
        if org_id:
            filter_dict["org_id"] = org_id
        return filter_dict
    
    async def get_storage_stats(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            db = get_sync_database()
            coll = db['chunks']
            scope_filter: Dict[str, Any] = {"org_id": org_id} if org_id else {}
            
            # Get basic stats
            total_chunks = coll.count_documents(scope_filter)
            total_embeddings = coll.count_documents({**scope_filter, "embedding": {"$exists": True, "$ne": []}})
            source_counts = {
                "CPG": coll.count_documents({**scope_filter, "source_type": "CPG"}),
                "EMR": coll.count_documents({**scope_filter, "source_type": "EMR"}),
                "LIT": coll.count_documents({**scope_filter, "source_type": "LIT"}),
            }
            if settings.CLOUD_MODE:
                active_embeddings = coll.count_documents(self._index_embedding_filter(org_id=org_id))
                stale_embeddings = coll.count_documents(self._stale_embedding_filter(org_id=org_id))
            else:
                active_embeddings = total_embeddings
                stale_embeddings = 0
            
            # Check index existence
            index_path = Path(settings.HNSW_INDEX_PATH)
            index_exists = index_path.exists()
            manifest_path = _index_manifest_path(index_path)
            index_manifest = self._read_index_manifest(index_path) if index_exists else None
            index_provenance_valid = self._manifest_matches_active_scope(index_manifest, org_id)
            index_provenance_error = None
            if index_exists and not index_manifest:
                index_provenance_error = "missing index provenance manifest"
            elif index_exists and not index_provenance_valid:
                index_provenance_error = "index provenance does not match active org or embedding space"
            index_size = index_path.stat().st_size if index_exists else None
            
            # Get last updated timestamp
            last_updated = None
            if total_chunks > 0:
                last_chunk = coll.find_one(
                    scope_filter,
                    sort=[("metadata.created_timestamp", -1)]
                )
                if last_chunk and 'metadata' in last_chunk:
                    last_updated = last_chunk['metadata'].get('created_timestamp')
            
            return {
                "total_chunks": total_chunks,
                "total_embeddings": total_embeddings,
                "source_counts": source_counts,
                "active_embeddings": active_embeddings,
                "stale_embeddings": stale_embeddings,
                "cloud_mode": settings.CLOUD_MODE,
                "active_embedding_model": settings.active_embedding_model(),
                "active_embedding_space": settings.active_embedding_space(),
                "active_embedding_dim": settings.active_embedding_dimension(),
                "embedding_refresh": self.get_embedding_refresh_status(),
                "index_exists": index_exists,
                "index_manifest_path": str(manifest_path),
                "index_manifest": index_manifest,
                "index_provenance_valid": index_provenance_valid,
                "index_provenance_error": index_provenance_error,
                "index_size": index_size,
                "last_updated": last_updated
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            raise
    
    async def clear_chunks(self, collection_name: str = "chunks") -> Dict[str, Any]:
        """Clear all chunks from storage."""
        try:
            db = get_sync_database()
            coll = db[collection_name]
            
            result = coll.delete_many({})
            
            return {
                "deleted_count": result.deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error clearing chunks: {e}")
            raise

    async def clear_benchmark_org(self, org_id: str, remove_indexes: bool = True) -> Dict[str, Any]:
        """Clear benchmark-scoped runtime data without touching other tenants.

        Motivation vs Logic: Benchmark reruns need a fresh corpus and traces, but
        deleting the whole database is unsafe in a shared development runtime.
        This reset deletes only the configured benchmark org and optionally
        removes global ANN files so they can be rebuilt from active embeddings.
        """
        try:
            db = get_sync_database()
            deleted: Dict[str, int] = {}
            for collection_name in ("chunks", "documents", "traces", "sessions"):
                result = db[collection_name].delete_many({"org_id": org_id})
                deleted[collection_name] = result.deleted_count

            removed_indexes = []
            if remove_indexes:
                for path_value in (
                    settings.HNSW_INDEX_PATH,
                    settings.HNSW_MAPPING_PATH,
                    settings.FAISS_INDEX_PATH,
                    settings.FAISS_MAPPING_PATH,
                    settings.TREE_INDEX_PATH,
                    settings.TREE_MAPPING_PATH,
                ):
                    path = Path(path_value)
                    if path.exists():
                        path.unlink()
                        removed_indexes.append(str(path))
                    manifest_path = _index_manifest_path(path)
                    if manifest_path.exists():
                        manifest_path.unlink()
                        removed_indexes.append(str(manifest_path))

            return {
                "org_id": org_id,
                "deleted": deleted,
                "removed_indexes": removed_indexes,
            }
        except Exception as e:
            logger.error(f"Error clearing benchmark org {org_id}: {e}")
            raise
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID."""
        try:
            db = get_sync_database()
            coll = db['chunks']
            
            chunk = coll.find_one({"chunk_id": chunk_id})
            
            if chunk:
                # Convert ObjectId to string
                chunk['_id'] = str(chunk['_id'])
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error getting chunk: {e}")
            raise
    
    async def list_chunks(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List chunks with optional filtering."""
        try:
            db = get_sync_database()
            coll = db['chunks']
            
            query = filters or {}
            cursor = coll.find(query).skip(skip).limit(limit)
            
            chunks = []
            for chunk in cursor:
                chunk['_id'] = str(chunk['_id'])
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error listing chunks: {e}")
            raise
    
    async def validate_storage(self) -> Dict[str, Any]:
        """Validate storage integrity."""
        try:
            db = get_sync_database()
            coll = db['chunks']
            
            # Get chunk count
            chunk_count = coll.count_documents({})
            
            # Check for chunks without embeddings
            chunks_without_embeddings = coll.count_documents({
                "embedding": {"$exists": False}
            })
            
            # Check for chunks with invalid embeddings
            chunks_with_invalid_embeddings = coll.count_documents({
                "embedding": {"$exists": True, "$size": 0}
            })
            
            # Check index existence and validity
            index_path = Path(settings.HNSW_INDEX_PATH)
            mapping_path = Path(settings.HNSW_MAPPING_PATH)
            
            index_exists = index_path.exists()
            mapping_exists = mapping_path.exists()
            
            index_valid = False
            if index_exists and mapping_exists:
                try:
                    # Try to load the index using HNSW builder
                    with open(mapping_path, 'r') as f:
                        mapping = json.load(f)
                    
                    if mapping:
                        # Get embedding dimension from first chunk
                        sample_chunk = coll.find_one(self._index_embedding_filter())
                        if sample_chunk:
                            embedding_dim = sample_chunk.get('embedding_dim', settings.active_embedding_dimension())
                            builder = HNSWIndexBuilder(embedding_dim)
                            index_valid = builder.load(str(index_path), str(mapping_path))
                except Exception as e:
                    logger.warning(f"Index validation failed: {e}")
            
            # Collect issues
            issues = []
            if chunks_without_embeddings > 0:
                issues.append(f"{chunks_without_embeddings} chunks without embeddings")
            if chunks_with_invalid_embeddings > 0:
                issues.append(f"{chunks_with_invalid_embeddings} chunks with invalid embeddings")
            if not index_exists:
                issues.append("HNSW index not found")
            if not mapping_exists:
                issues.append("HNSW mapping not found")
            if index_exists and not index_valid:
                issues.append("HNSW index is corrupted or invalid")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "chunk_count": chunk_count,
                "index_exists": index_exists,
                "index_valid": index_valid
            }
            
        except Exception as e:
            logger.error(f"Error validating storage: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
