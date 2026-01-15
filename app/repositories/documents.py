"""Document repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.repositories.base import BaseRepository
from app.models.medswin import Document, SourceType
import logging

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for documents with org-aware partitioning."""
    
    def __init__(self):
        """Initialize document repository."""
        super().__init__("documents")
    
    async def create_indexes(self):
        """Create indexes for documents collection."""
        await self.collection.create_index("doc_id", unique=True)
        await self.collection.create_index([("org_id", 1), ("doc_id", 1)])
        await self.collection.create_index([("org_id", 1), ("source_type", 1)])
        await self.collection.create_index([("org_id", 1), ("patient_id", 1)])
        await self.collection.create_index([("org_id", 1), ("effective_date", -1)])
        logger.info("Documents collection indexes created")
    
    async def create(self, document: Document, org_id: str) -> Dict[str, Any]:
        """Create a document."""
        data = document.model_dump()
        data = self._ensure_org_id(data, org_id)
        result = await self.collection.insert_one(data)
        return {"doc_id": document.doc_id, "inserted_id": str(result.inserted_id)}
    
    async def create_many(self, documents: List[Document], org_id: str) -> List[Dict[str, Any]]:
        """Create multiple documents."""
        data_list = [self._ensure_org_id(doc.model_dump(), org_id) for doc in documents]
        result = await self.collection.insert_many(data_list)
        return [{"doc_id": doc.doc_id, "inserted_id": str(id)} 
                for doc, id in zip(documents, result.inserted_ids)]
    
    async def get_by_id(self, doc_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return await self.collection.find_one({"doc_id": doc_id, "org_id": org_id})
    
    async def get_by_ids(self, doc_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        """Get documents by IDs."""
        cursor = self.collection.find({"doc_id": {"$in": doc_ids}, "org_id": org_id})
        return await cursor.to_list(length=None)
    
    async def get_by_source_type(self, source_type: SourceType, org_id: str,
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents by source type."""
        cursor = self.collection.find({"source_type": source_type.value, "org_id": org_id})
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)
    
    async def get_by_patient_id(self, patient_id: str, org_id: str) -> List[Dict[str, Any]]:
        """Get documents by patient ID."""
        cursor = self.collection.find({"patient_id": patient_id, "org_id": org_id})
        return await cursor.to_list(length=None)
    
    async def update(self, doc_id: str, org_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document."""
        result = await self.collection.update_one(
            {"doc_id": doc_id, "org_id": org_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def delete(self, doc_id: str, org_id: str) -> bool:
        """Delete a document."""
        result = await self.collection.delete_one({"doc_id": doc_id, "org_id": org_id})
        return result.deleted_count > 0
    
    async def count_by_org(self, org_id: str) -> int:
        """Count documents for an organization."""
        return await self.collection.count_documents({"org_id": org_id})

