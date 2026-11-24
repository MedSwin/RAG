"""Trace repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.repositories.base import BaseRepository
from app.models.medswin import AuditTrace
from app.services.medswin.governance import redact_stored_trace
import logging

logger = logging.getLogger(__name__)


class TraceRepository(BaseRepository):
    """Repository for audit traces with org-aware partitioning."""
    
    def __init__(self):
        """Initialize trace repository."""
        super().__init__("traces")
    
    async def create_indexes(self):
        """Create indexes for traces collection."""
        await self.collection.create_index("trace_id", unique=True)
        await self.collection.create_index([("org_id", 1), ("trace_id", 1)])
        await self.collection.create_index([("org_id", 1), ("session_id", 1)])
        await self.collection.create_index([("org_id", 1), ("user_id", 1)])
        await self.collection.create_index([("org_id", 1), ("created_at", -1)])
        await self.collection.create_index([("org_id", 1), ("patient_id", 1)])
        logger.info("Traces collection indexes created")
    
    async def create(self, trace: AuditTrace, org_id: str) -> Dict[str, Any]:
        """Create a trace."""
        data = trace.model_dump()
        data = self._ensure_org_id(data, org_id)
        if settings.LOG_REDACT_PHI:
            data = redact_stored_trace(data)
        result = await self.collection.insert_one(data)
        return {"trace_id": trace.trace_id, "inserted_id": str(result.inserted_id)}
    
    async def get_by_id(self, trace_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        return await self.collection.find_one({"trace_id": trace_id, "org_id": org_id})
    
    async def get_by_session_id(
        self,
        session_id: str,
        org_id: str,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get traces for a session when this clinician owns them."""
        if not user_id:
            return []
        cursor = self.collection.find(
            {"session_id": session_id, "org_id": org_id, "user_id": user_id}
        ).sort("created_at", -1)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)
    
    async def get_by_user_id(self, user_id: str, org_id: str,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get traces by user ID."""
        cursor = self.collection.find({"user_id": user_id, "org_id": org_id}).sort("created_at", -1)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)
    
    async def update(self, trace_id: str, org_id: str, updates: Dict[str, Any], user_id: str) -> bool:
        """Update a trace when this clinician owns it."""
        if not user_id or not updates:
            return False
        if any(not isinstance(key, str) or "." in key or key.startswith("$") for key in updates):
            return False
        if {"trace_id", "org_id", "user_id", "session_id", "patient_id"} & set(updates):
            return False
        result = await self.collection.update_one(
            {"trace_id": trace_id, "org_id": org_id, "user_id": user_id},
            {"$set": updates},
        )
        return result.matched_count > 0
    
    async def count_by_org(self, org_id: str) -> int:
        """Count traces for an organization."""
        return await self.collection.count_documents({"org_id": org_id})

