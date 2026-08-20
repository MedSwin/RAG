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
        """Create or replace a chunk within its tenant natural key."""
        data = self._ensure_org_id(chunk.model_dump(), org_id)
        result = await self.collection.replace_one(
            {"org_id": org_id, "chunk_id": chunk.chunk_id},
            data,
            upsert=True,
        )
        return {
            "chunk_id": chunk.chunk_id,
            "inserted_id": str(result.upserted_id) if result.upserted_id else chunk.chunk_id,
            "upserted": bool(result.upserted_id),
        }

    async def create_many(self, chunks: List[Chunk], org_id: str) -> List[Dict[str, Any]]:
        """Create or replace multiple chunks inside one tenant."""
        if not chunks:
            return []
        operations = []
        for chunk in chunks:
            data = self._ensure_org_id(chunk.model_dump(), org_id)
            operations.append(
                ReplaceOne(
                    {"org_id": org_id, "chunk_id": chunk.chunk_id},
                    data,
                    upsert=True,
                )
            )
        result = await self.collection.bulk_write(operations, ordered=False)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "inserted_id": str(result.upserted_ids.get(idx, chunk.chunk_id)),
                "upserted": idx in result.upserted_ids,
            }
            for idx, chunk in enumerate(chunks)
        ]

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

    async def get_by_patient_id(self, patient_id: str, org_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"patient_id": patient_id, "org_id": org_id})
        return await cursor.to_list(length=None)

    async def update(self, chunk_id: str, org_id: str, updates: Dict[str, Any]) -> bool:
        result = await self.collection.update_one(
            {"chunk_id": chunk_id, "org_id": org_id},
            {"$set": updates},
        )
        return result.modified_count > 0

    async def delete(self, chunk_id: str, org_id: str) -> bool:
        result = await self.collection.delete_one({"chunk_id": chunk_id, "org_id": org_id})
        return result.deleted_count > 0

    async def count_by_org(self, org_id: str) -> int:
        return await self.collection.count_documents({"org_id": org_id})
