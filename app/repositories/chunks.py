"""Chunk repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.repositories.base import BaseRepository
from app.models.medswin import Chunk, SourceType
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)


class ChunkRepository(BaseRepository):
    """Repository for chunks with org-aware partitioning."""
    
    def __init__(self):
        """Initialize chunk repository."""
        super().__init__("chunks")
    
    async def create_indexes(self):
        """Create indexes for chunks collection."""
        await self.collection.create_index("chunk_id", unique=True)
        await self.collection.create_index([("org_id", 1), ("chunk_id", 1)])
        await self.collection.create_index([("org_id", 1), ("doc_id", 1)])
        await self.collection.create_index([("org_id", 1), ("source_type", 1)])
        await self.collection.create_index([("org_id", 1), ("patient_id", 1)])
        await self.collection.create_index([("text", "text")])
        logger.info("Chunks collection indexes created")
    
    async def create(self, chunk: Chunk, org_id: str) -> Dict[str, Any]:
        """Create a chunk."""
        data = chunk.model_dump()
        data = self._ensure_org_id(data, org_id)
        result = await self.collection.insert_one(data)
        return {"chunk_id": chunk.chunk_id, "inserted_id": str(result.inserted_id)}
    
    async def create_many(self, chunks: List[Chunk], org_id: str) -> List[Dict[str, Any]]:
        """Create multiple chunks."""
        data_list = [self._ensure_org_id(chunk.model_dump(), org_id) for chunk in chunks]
        result = await self.collection.insert_many(data_list)
        return [{"chunk_id": chunk.chunk_id, "inserted_id": str(id)} 
                for chunk, id in zip(chunks, result.inserted_ids)]
    
    async def get_by_id(self, chunk_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        return await self.collection.find_one({"chunk_id": chunk_id, "org_id": org_id})
    
    async def get_by_ids(self, chunk_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        """Get chunks by IDs."""
        cursor = self.collection.find({"chunk_id": {"$in": chunk_ids}, "org_id": org_id})
        return await cursor.to_list(length=None)
    
    async def get_by_doc_id(self, doc_id: str, org_id: str) -> List[Dict[str, Any]]:
        """Get chunks by document ID."""
        cursor = self.collection.find({"doc_id": doc_id, "org_id": org_id})
        return await cursor.to_list(length=None)
    
    async def get_by_source_type(self, source_type: SourceType, org_id: str, 
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chunks by source type."""
        cursor = self.collection.find({"source_type": source_type.value, "org_id": org_id})
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)
    
    async def get_by_patient_id(self, patient_id: str, org_id: str) -> List[Dict[str, Any]]:
        """Get chunks by patient ID."""
        cursor = self.collection.find({"patient_id": patient_id, "org_id": org_id})
        return await cursor.to_list(length=None)
    
    async def update(self, chunk_id: str, org_id: str, updates: Dict[str, Any]) -> bool:
        """Update a chunk."""
        result = await self.collection.update_one(
            {"chunk_id": chunk_id, "org_id": org_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def delete(self, chunk_id: str, org_id: str) -> bool:
        """Delete a chunk."""
        result = await self.collection.delete_one({"chunk_id": chunk_id, "org_id": org_id})
        return result.deleted_count > 0
    
    async def count_by_org(self, org_id: str) -> int:
        """Count chunks for an organization."""
        return await self.collection.count_documents({"org_id": org_id})

