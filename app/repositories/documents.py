"""Document repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.repositories.base import BaseRepository
from app.models.medswin import Document, SourceType
from pymongo import ReplaceOne
import logging

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for documents with org-aware partitioning.

    ``doc_id`` is a source identifier, not a globally unique tenant identifier.
    The natural persistence key is therefore ``(org_id, doc_id)``. This matters
    for benchmark corpora such as PMC, whose source IDs can legitimately appear
    in more than one organization/workspace.
    """

    def __init__(self):
        super().__init__("documents")

    async def create_indexes(self):
        """Create tenant-safe indexes for documents."""
        # Drop the historical global-unique doc_id index when it exists. Keeping
        # it would make a valid second-tenant ingest fail even after adding the
        # correct compound unique index.
        try:
            info = await self.collection.index_information()
            for name, spec in info.items():
                keys = spec.get("key") or []
                if keys == [("doc_id", 1)] and spec.get("unique"):
                    await self.collection.drop_index(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not inspect/drop legacy document doc_id index: %s", exc)

        await self.collection.create_index([("org_id", 1), ("doc_id", 1)], unique=True)
        await self.collection.create_index([("org_id", 1), ("source_type", 1)])
        await self.collection.create_index([("org_id", 1), ("patient_id", 1)])
        await self.collection.create_index([("org_id", 1), ("effective_date", -1)])
        logger.info("Documents collection tenant-safe indexes created")

    async def create(self, document: Document, org_id: str) -> Dict[str, Any]:
        """Create or replace a document within its tenant natural key."""
        data = self._ensure_org_id(document.model_dump(), org_id)
        result = await self.collection.replace_one(
            {"org_id": org_id, "doc_id": document.doc_id},
            data,
            upsert=True,
        )
        return {
            "doc_id": document.doc_id,
            "inserted_id": str(result.upserted_id) if result.upserted_id else document.doc_id,
            "upserted": bool(result.upserted_id),
        }

    async def create_many(self, documents: List[Document], org_id: str) -> List[Dict[str, Any]]:
        """Create or replace multiple documents inside one tenant."""
        if not documents:
            return []
        operations = []
        for doc in documents:
            data = self._ensure_org_id(doc.model_dump(), org_id)
            operations.append(
                ReplaceOne(
                    {"org_id": org_id, "doc_id": doc.doc_id},
                    data,
                    upsert=True,
                )
            )
        result = await self.collection.bulk_write(operations, ordered=False)
        return [
            {
                "doc_id": doc.doc_id,
                "inserted_id": str(result.upserted_ids.get(idx, doc.doc_id)),
                "upserted": idx in result.upserted_ids,
            }
            for idx, doc in enumerate(documents)
        ]

    async def get_by_id(self, doc_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by tenant-scoped source ID."""
        return await self.collection.find_one({"doc_id": doc_id, "org_id": org_id})

    async def get_by_ids(self, doc_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"doc_id": {"$in": doc_ids}, "org_id": org_id})
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

    async def update(self, doc_id: str, org_id: str, updates: Dict[str, Any]) -> bool:
        result = await self.collection.update_one(
            {"doc_id": doc_id, "org_id": org_id},
            {"$set": updates},
        )
        return result.modified_count > 0

    async def delete(self, doc_id: str, org_id: str) -> bool:
        result = await self.collection.delete_one({"doc_id": doc_id, "org_id": org_id})
        return result.deleted_count > 0

    async def count_by_org(self, org_id: str) -> int:
        return await self.collection.count_documents({"org_id": org_id})
