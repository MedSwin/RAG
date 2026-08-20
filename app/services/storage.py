try:
    import pymongo
except ModuleNotFoundError:
    pymongo = None
try:
    from pymongo import MongoClient, UpdateOne
except ModuleNotFoundError:
    MongoClient = UpdateOne = None
import hashlib
import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
    """Derive a deterministic corpus signature from active index contents."""
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
    """Service for managing data storage and ordinary (non-publication) indexes.

    The complete-TREC publication path uses its dedicated streaming builder.
    This service deliberately refuses to materialize an unbounded vector corpus
    into Python memory so an admin endpoint cannot accidentally OOM a live API.
    """

    def __init__(self):
        # ThreadPoolExecutor creates worker threads lazily, but retaining it only
        # when a blocking operation is actually used avoids health/stats callers
        # allocating executor objects on every request.
        self.executor: Optional[ThreadPoolExecutor] = None

    def _executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="medswin-storage")
        return self.executor

    def _write_index_manifest(self, index_path: str, manifest: Dict[str, Any]) -> str:
        manifest_path = _index_manifest_path(index_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
        return str(manifest_path)

    def _read_index_manifest(self, index_path: str | Path) -> Dict[str, Any] | None:
        manifest_path = _index_manifest_path(index_path)
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
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
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Idempotently store tenant-scoped chunks in MongoDB."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor(),
                self._store_chunks_sync,
                chunks,
                collection_name,
                batch_size,
            )
        except Exception as exc:
            logger.error("Error storing chunks: %s", exc)
            raise

    def _store_chunks_sync(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Synchronous tenant-safe chunk storage function."""
        if MongoClient is None or UpdateOne is None:
            raise RuntimeError("pymongo is required for chunk storage")
        batch_size = max(1, int(batch_size))
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DATABASE]
        coll = db[collection_name]

        success_count = 0
        skipped_count = 0
        failed_count = 0
        failed_chunks: List[Dict[str, Any]] = []
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                operations = []
                operation_chunks: List[Dict[str, Any]] = []
                for raw in batch:
                    chunk = dict(raw)
                    chunk_id = str(chunk.get("chunk_id") or "").strip()
                    org_id = str(chunk.get("org_id") or "").strip()
                    if collection_name == "chunks" and (not chunk_id or not org_id):
                        failed_chunks.append(chunk)
                        failed_count += 1
                        continue
                    if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                        created = chunk["metadata"].get("created_timestamp")
                        if isinstance(created, datetime) and created.tzinfo is None:
                            chunk["metadata"]["created_timestamp"] = created.replace(tzinfo=timezone.utc)
                    natural_key = {"chunk_id": chunk_id}
                    if org_id:
                        natural_key["org_id"] = org_id
                    operations.append(UpdateOne(natural_key, {"$setOnInsert": chunk}, upsert=True))
                    operation_chunks.append(chunk)

                if not operations:
                    continue
                try:
                    result = coll.bulk_write(operations, ordered=False)
                    inserted = len(result.upserted_ids)
                    success_count += inserted
                    skipped_count += len(operations) - inserted
                    logger.info(
                        "Stored batch %s: inserted=%s existing=%s",
                        start // batch_size + 1,
                        inserted,
                        len(operations) - inserted,
                    )
                except pymongo.errors.BulkWriteError as exc:
                    logger.error("Chunk batch write failed: %s", exc.details)
                    failed_chunks.extend(operation_chunks)
                    failed_count += len(operation_chunks)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Unexpected chunk storage error: %s", exc)
                    failed_chunks.extend(operation_chunks)
                    failed_count += len(operation_chunks)
        finally:
            client.close()

        if failed_chunks:
            failed_path = Path(settings.DATA_DIR) / "failed_chunks.json"
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(json.dumps(failed_chunks, default=str, indent=2), encoding="utf-8")
            logger.warning("%s failed chunks saved to %s", len(failed_chunks), failed_path)

        return {
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "total_chunks": len(chunks),
        }

    async def build_hnsw_index_async(
        self,
        index_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        force_rebuild: bool = False,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an ordinary HNSW index asynchronously."""
        try:
            index_path = index_path or settings.HNSW_INDEX_PATH
            mapping_path = mapping_path or settings.HNSW_MAPPING_PATH
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

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor(),
                self._build_hnsw_index_sync,
                index_path,
                mapping_path,
                org_id,
            )
        except Exception as exc:
            logger.error("Error building HNSW index: %s", exc)
            raise

    async def refresh_cloud_embeddings(
        self,
        batch_size: Optional[int] = None,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Refresh stale chunk embeddings and rebuild the active index.

        Stored passages are documents, so Cohere-native cloud calls explicitly
        use document intent rather than relying on the online-query default.
        """
        batch_size = batch_size or settings.BATCH_SIZE
        batch_cap = settings.CLOUD_EMBED_BATCH_SIZE if settings.CLOUD_MODE else settings.BATCH_SIZE
        batch_size = max(1, min(int(batch_size), int(batch_cap)))
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
                    embeddings = await client.embed(
                        texts,
                        input_type="document" if settings.CLOUD_MODE else None,
                    )
                    if len(embeddings) != len(chunks):
                        raise RuntimeError(
                            f"Embedding service returned {len(embeddings)} vectors for {len(chunks)} chunks"
                        )
                    for chunk, embedding in zip(chunks, embeddings):
                        await coll.update_one(
                            {"_id": chunk["_id"]},
                            {"$set": {
                                "embedding": embedding.tolist(),
                                "embedding_model": settings.active_embedding_model(),
                                "embedding_dim": int(len(embedding)),
                                "embedding_space": settings.active_embedding_space(),
                                "embedding_updated_at": datetime.now(timezone.utc),
                            }},
                        )
                        EMBEDDING_REFRESH_STATUS["updated"] += 1
                    if settings.CLOUD_MODE and settings.CLOUD_EMBED_BATCH_DELAY_S > 0:
                        await asyncio.sleep(settings.CLOUD_EMBED_BATCH_DELAY_S)
            finally:
                await client.close()

            index_result = await self.build_hnsw_index_async(force_rebuild=True, org_id=org_id)
            if not index_result.get("success"):
                raise RuntimeError(index_result.get("message") or "HNSW rebuild failed after embedding refresh")
            EMBEDDING_REFRESH_STATUS.update({
                "running": False,
                "ready": True,
                "completed_at": datetime.now(timezone.utc),
            })
        except Exception as exc:
            logger.error("Cloud embedding refresh failed: %s", exc, exc_info=True)
            EMBEDDING_REFRESH_STATUS.update({
                "running": False,
                "ready": False,
                "error": str(exc),
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
                {"embedding_model": {"$ne": settings.active_embedding_model()}},
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
        index_type: IndexType = IndexType.HNSW,
    ) -> Dict[str, Any]:
        """Build an in-memory index only for bounded ordinary corpora."""
        try:
            db = get_sync_database()
            coll = db["chunks"]
            active_filter = self._index_embedding_filter(org_id=org_id)
            vector_count = int(coll.count_documents(active_filter))
            max_vectors = max(1, int(os.getenv("STORAGE_IN_MEMORY_INDEX_MAX_VECTORS", "250000")))
            if vector_count > max_vectors:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "total_vectors": 0,
                    "message": (
                        f"Refusing in-memory index build for {vector_count:,} vectors; "
                        f"limit is {max_vectors:,}. Use the streaming full-corpus builder for large corpora."
                    ),
                }

            chunks = list(coll.find(
                active_filter,
                {"chunk_id": 1, "doc_id": 1, "source_type": 1, "embedding": 1, "embedding_dim": 1, "metadata": 1},
            ))
            if not chunks:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "total_vectors": 0,
                    "message": "No chunks with embeddings found",
                }

            embedding_dim = chunks[0].get("embedding_dim", settings.active_embedding_dimension())
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
                embeddings.append(chunk["embedding"])
                chunk_ids.append(chunk["chunk_id"])
            if not embeddings:
                return {
                    "success": False,
                    "index_path": index_path,
                    "mapping_path": mapping_path,
                    "manifest_path": str(_index_manifest_path(index_path)),
                    "total_vectors": 0,
                    "message": "No chunks with active-dimension embeddings found",
                }

            strategy_manager = IndexStrategyManager()
            if index_type == IndexType.HNSW:
                config = strategy_manager.get_index_config(IndexStrategy.HNSW_ONLY, embedding_dim, len(chunks))
                builder = HNSWIndexBuilder(embedding_dim, config.get("hnsw"))
            elif index_type == IndexType.FAISS_IVF:
                config = strategy_manager.get_index_config(IndexStrategy.FAISS_ONLY, embedding_dim, len(chunks))
                builder = FAISSIndexBuilder(embedding_dim, config.get("faiss_ivf"))
            elif index_type == IndexType.FAISS_TREE:
                config = strategy_manager.get_index_config(IndexStrategy.TREE_ONLY, embedding_dim, len(chunks))
                builder = TreeIndexBuilder(embedding_dim, config.get("faiss_tree"))
            else:
                config = strategy_manager.get_index_config(IndexStrategy.HNSW_ONLY, embedding_dim, len(chunks))
                builder = HNSWIndexBuilder(embedding_dim, config.get("hnsw"))

            result = builder.build(embeddings, chunk_ids, index_path, mapping_path)
            if not result.get("success"):
                return result
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
                "doc_ids": sorted({str(chunk.get("doc_id")) for chunk in chunks if chunk.get("doc_id")}),
                "corpus_signature": _corpus_signature(chunk_ids, org_id or "all", settings.active_embedding_space()),
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = self._write_index_manifest(index_path, manifest)
            result["manifest_path"] = manifest_path
            result["index_manifest"] = manifest
            logger.info("%s index built for %s: %s", index_type.value.upper(), org_id or "all orgs", result["message"])
            return result
        except Exception as exc:
            logger.error("Error building index: %s", exc)
            return {
                "success": False,
                "index_path": index_path,
                "mapping_path": mapping_path,
                "manifest_path": str(_index_manifest_path(index_path)),
                "total_vectors": 0,
                "message": f"Index building failed: {exc}",
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
        """Get storage statistics without materializing a million-ID corpus."""
        try:
            db = get_sync_database()
            coll = db["chunks"]
            scope_filter: Dict[str, Any] = {"org_id": org_id} if org_id else {}
            total_chunks = int(coll.count_documents(scope_filter))
            total_embeddings = int(coll.count_documents({**scope_filter, "embedding": {"$exists": True, "$ne": []}}))
            source_counts = {
                "CPG": int(coll.count_documents({**scope_filter, "source_type": "CPG"})),
                "EMR": int(coll.count_documents({**scope_filter, "source_type": "EMR"})),
                "LIT": int(coll.count_documents({**scope_filter, "source_type": "LIT"})),
            }
            if settings.CLOUD_MODE:
                active_embeddings = int(coll.count_documents(self._index_embedding_filter(org_id=org_id)))
                stale_embeddings = int(coll.count_documents(self._stale_embedding_filter(org_id=org_id)))
            else:
                active_embeddings = total_embeddings
                stale_embeddings = 0

            index_path = Path(settings.HNSW_INDEX_PATH)
            index_exists = index_path.exists()
            manifest_path = _index_manifest_path(index_path)
            index_manifest = self._read_index_manifest(index_path) if index_exists else None
            index_provenance_valid = self._manifest_matches_active_scope(index_manifest, org_id)
            if index_manifest:
                index_provenance_valid = (
                    index_provenance_valid
                    and int(index_manifest.get("total_vectors") or 0) == active_embeddings
                )
            index_provenance_error = None
            if index_exists and not index_manifest:
                index_provenance_error = "missing index provenance manifest"
            elif index_exists and not index_provenance_valid:
                index_provenance_error = "index provenance/vector count does not match active org or embedding space"

            active_doc_ids = list((index_manifest or {}).get("doc_ids") or [])
            # Old ordinary manifests may contain large doc-id arrays. API stats
            # are diagnostic, not a corpus export, so bound response size.
            active_doc_ids = [str(value) for value in active_doc_ids[:1000]]
            index_size = index_path.stat().st_size if index_exists else None

            last_updated = None
            if total_chunks > 0:
                last_chunk = coll.find_one(scope_filter, sort=[("metadata.created_timestamp", -1)])
                if last_chunk and "metadata" in last_chunk:
                    last_updated = last_chunk["metadata"].get("created_timestamp")

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
                "active_doc_ids": active_doc_ids,
                "last_updated": last_updated,
            }
        except Exception as exc:
            logger.error("Error getting storage stats: %s", exc)
            raise

    async def clear_chunks(self, collection_name: str = "chunks", org_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear chunks, tenant-scoped when org_id is supplied."""
        try:
            db = get_sync_database()
            coll = db[collection_name]
            result = coll.delete_many({"org_id": org_id} if org_id else {})
            return {"deleted_count": int(result.deleted_count), "org_id": org_id}
        except Exception as exc:
            logger.error("Error clearing chunks: %s", exc)
            raise

    async def clear_benchmark_org(self, org_id: str, remove_indexes: bool = True) -> Dict[str, Any]:
        """Clear benchmark-scoped runtime data without touching other tenants."""
        try:
            db = get_sync_database()
            deleted: Dict[str, int] = {}
            for collection_name in ("chunks", "documents", "traces", "sessions"):
                result = db[collection_name].delete_many({"org_id": org_id})
                deleted[collection_name] = int(result.deleted_count)

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
            return {"org_id": org_id, "deleted": deleted, "removed_indexes": removed_indexes}
        except Exception as exc:
            logger.error("Error clearing benchmark org %s: %s", org_id, exc)
            raise

    async def get_chunk(self, chunk_id: str, org_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a specific chunk, tenant-scoped when org_id is supplied."""
        try:
            db = get_sync_database()
            coll = db["chunks"]
            query: Dict[str, Any] = {"chunk_id": chunk_id}
            if org_id:
                query["org_id"] = org_id
            chunk = coll.find_one(query)
            if chunk:
                chunk["_id"] = str(chunk["_id"])
            return chunk
        except Exception as exc:
            logger.error("Error getting chunk: %s", exc)
            raise

    async def list_chunks(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List chunks with bounded pagination and optional filters."""
        try:
            db = get_sync_database()
            coll = db["chunks"]
            query = filters or {}
            safe_limit = max(1, min(int(limit), 1000))
            cursor = coll.find(query).skip(max(0, int(skip))).limit(safe_limit)
            chunks = []
            for chunk in cursor:
                chunk["_id"] = str(chunk["_id"])
                chunks.append(chunk)
            return chunks
        except Exception as exc:
            logger.error("Error listing chunks: %s", exc)
            raise

    async def validate_storage(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate storage/index integrity for an optional tenant scope."""
        try:
            db = get_sync_database()
            coll = db["chunks"]
            scope: Dict[str, Any] = {"org_id": org_id} if org_id else {}
            chunk_count = int(coll.count_documents(scope))
            chunks_without_embeddings = int(coll.count_documents({**scope, "embedding": {"$exists": False}}))
            chunks_with_invalid_embeddings = int(coll.count_documents({**scope, "embedding": {"$exists": True, "$size": 0}}))

            index_path = Path(settings.HNSW_INDEX_PATH)
            mapping_path = Path(settings.HNSW_MAPPING_PATH)
            index_exists = index_path.exists()
            mapping_exists = mapping_path.exists()
            index_valid = False
            if index_exists and mapping_exists and HNSWIndexBuilder is not None:
                try:
                    sample_chunk = coll.find_one(self._index_embedding_filter(org_id=org_id))
                    if sample_chunk:
                        embedding_dim = sample_chunk.get("embedding_dim", settings.active_embedding_dimension())
                        builder = HNSWIndexBuilder(embedding_dim)
                        index_valid = builder.load(str(index_path), str(mapping_path))
                        if hasattr(builder.mapping, "close"):
                            builder.mapping.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Index validation failed: %s", exc)

            issues = []
            if chunks_without_embeddings > 0:
                issues.append(f"{chunks_without_embeddings} chunks without embeddings")
            if chunks_with_invalid_embeddings > 0:
                issues.append(f"{chunks_with_invalid_embeddings} chunks with invalid embeddings")
            if not index_exists:
                issues.append("HNSW index not found")
            if not mapping_exists:
                issues.append("HNSW mapping not found")
            if index_exists and mapping_exists and not index_valid:
                issues.append("HNSW index is corrupted, incompatible, or invalid")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "org_id": org_id,
                "chunk_count": chunk_count,
                "index_exists": index_exists,
                "index_valid": index_valid,
            }
        except Exception as exc:
            logger.error("Error validating storage: %s", exc)
            raise

    def cleanup(self):
        """Cleanup worker resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
