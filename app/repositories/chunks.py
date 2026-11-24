"""Chunk repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.repositories.base import BaseRepository
from app.models.medswin import Chunk, SourceType
from pymongo import ReplaceOne
import logging

logger = logging.getLogger(__name__)


class ChunkRepository(BaseRepository):
    """Repository for chunks with tenant-scoped natural identities.

    Lexical retrieval is implemented by the runtime BM25/SQLite-FTS layer, not
    Mongo ``$text``. We intentionally do not maintain a second text index over
    millions of chunk bodies because it duplicates the retrieval corpus and can
    make complete-TREC ingestion substantially more expensive.
    """

    def __init__(self):
        super().__init__("chunks")

    async def create_indexes(self):
        """Create tenant-safe indexes for chunks and remove legacy text indexes."""
        try:
            info = await self.collection.index_information()
            for name, spec in info.items():
                keys = spec.get("key") or []
                if keys == [("chunk_id", 1)] and spec.get("unique"):
                    await self.collection.drop_index(name)
                    continue
                if any(str(kind) == "text" for _field, kind in keys):
                    await self.collection.drop_index(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not migrate legacy chunk indexes: %s", exc)
            raise

        await self.collection.create_index([("org_id", 1), ("chunk_id", 1)], unique=True)
        await self.collection.create_index([("org_id", 1), ("doc_id", 1)])
        await self.collection.create_index([("org_id", 1), ("source_type", 1)])
        await self.collection.create_index([("org_id", 1), ("patient_id", 1)])
        logger.info("Chunks collection tenant-safe indexes created")

    async def create(self, chunk: Chunk, org_id: str) -> Dict[str, Any]:
        """Create or replace a chunk without moving its patient."""
        data = self._ensure_org_id(chunk.model_dump(), org_id)
        existing = await self.collection.find_one(
            {"org_id": org_id, "chunk_id": chunk.chunk_id},
            {"patient_id": 1, "source_type": 1, "doc_id": 1},
        )
        if self._identity_changed(existing, data, ("patient_id", "source_type", "doc_id")):
            return {
                "chunk_id": chunk.chunk_id,
                "inserted_id": chunk.chunk_id,
                "upserted": False,
                "rejected": True,
            }
        result = await self.collection.replace_one(
            {"org_id": org_id, "chunk_id": chunk.chunk_id},
            data,
            upsert=True,
        )
        return {
            "chunk_id": chunk.chunk_id,
            "inserted_id": str(result.upserted_id) if result.upserted_id else chunk.chunk_id,
            "upserted": bool(result.upserted_id),
            "rejected": False,
        }

    async def create_many(self, chunks: List[Chunk], org_id: str) -> List[Dict[str, Any]]:
        """Create or replace multiple chunks without moving a patient."""
        if not chunks:
            return []
        cursor = self.collection.find(
            {"org_id": org_id, "chunk_id": {"$in": [chunk.chunk_id for chunk in chunks]}},
            {"chunk_id": 1, "patient_id": 1, "source_type": 1, "doc_id": 1},
        )
        existing_rows = await cursor.to_list(length=None)
        existing = {row.get("chunk_id"): row for row in existing_rows}
        operations = []
        kept: List[tuple] = []
        rejected = set()
        for idx, chunk in enumerate(chunks):
            data = self._ensure_org_id(chunk.model_dump(), org_id)
            if self._identity_changed(existing.get(chunk.chunk_id), data, ("patient_id", "source_type", "doc_id")):
                rejected.add(idx)
                continue
            operations.append(ReplaceOne({"org_id": org_id, "chunk_id": chunk.chunk_id}, data, upsert=True))
            kept.append((idx, chunk))
        result = await self.collection.bulk_write(operations, ordered=False) if operations else None
        rows: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        for write_idx, (idx, chunk) in enumerate(kept):
            upserted_id = result.upserted_ids.get(write_idx) if result else None
            rows[idx] = {
                "chunk_id": chunk.chunk_id,
                "inserted_id": str(upserted_id) if upserted_id else chunk.chunk_id,
                "upserted": bool(upserted_id),
                "rejected": False,
            }
        for idx in rejected:
            chunk = chunks[idx]
            rows[idx] = {
                "chunk_id": chunk.chunk_id,
                "inserted_id": chunk.chunk_id,
                "upserted": False,
                "rejected": True,
            }
        return [row for row in rows if row is not None]

    async def get_by_id(self, chunk_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        return await self.collection.find_one({"chunk_id": chunk_id, "org_id": org_id})

    async def get_by_ids(self, chunk_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"chunk_id": {"$in": chunk_ids}, "org_id": org_id})
        return await cursor.to_list(length=None)

    async def get_by_doc_id(self, doc_id: str, org_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"doc_id": doc_id, "org_id": org_id})
        return await cursor.to_list(length=None)

    async def get_by_source_type(
        self,
        source_type: SourceType,
        org_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"source_type": source_type.value, "org_id": org_id})
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)

    async def get_by_patient_id(
        self,
        patient_id: str,
        org_id: str,
        *,
        source_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"patient_id": patient_id, "org_id": org_id}
        if source_type:
            query["source_type"] = source_type
        cursor = self.collection.find(query).sort([("metadata.created_timestamp", -1)])
        bounded = max(1, int(limit)) if limit else None
        if bounded:
            cursor = cursor.limit(bounded)
        return await cursor.to_list(length=bounded)

    async def update(self, chunk_id: str, org_id: str, updates: Dict[str, Any]) -> bool:
        """Update a chunk without moving its identity or its patient."""
        if not updates:
            return False
        if any(not isinstance(key, str) or "." in key or key.startswith("$") for key in updates):
            return False
        if {"chunk_id", "doc_id", "org_id", "patient_id", "source_type"} & set(updates):
            return False
        result = await self.collection.update_one(
            {"chunk_id": chunk_id, "org_id": org_id},
            {"$set": updates},
        )
        return result.matched_count > 0

    async def delete(self, chunk_id: str, org_id: str) -> bool:
        result = await self.collection.delete_one({"chunk_id": chunk_id, "org_id": org_id})
        return result.deleted_count > 0

    async def count_by_org(self, org_id: str) -> int:
        return await self.collection.count_documents({"org_id": org_id})
