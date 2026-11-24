"""Session repository with org-aware partitioning."""

from typing import List, Optional, Dict, Any
from app.repositories.base import BaseRepository
from app.models.medswin import Session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionRepository(BaseRepository):
    """Repository for sessions with org-aware partitioning."""
    
    def __init__(self):
        """Initialize session repository."""
        super().__init__("sessions")
    
    async def create_indexes(self):
        """Create indexes for sessions collection."""
        await self.collection.create_index("session_id", unique=True)
        await self.collection.create_index([("org_id", 1), ("session_id", 1)])
        await self.collection.create_index([("org_id", 1), ("user_id", 1)])
        await self.collection.create_index([("org_id", 1), ("last_active", -1)])
        logger.info("Sessions collection indexes created")
    
    async def create(self, session: Session, org_id: str) -> Dict[str, Any]:
        """Create a session."""
        data = session.model_dump()
        data = self._ensure_org_id(data, org_id)
        result = await self.collection.insert_one(data)
        return {"session_id": session.session_id, "inserted_id": str(result.inserted_id)}
    
    async def get_by_id(self, session_id: str, org_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return await self.collection.find_one({"session_id": session_id, "org_id": org_id})
    
    async def get_by_user_id(self, user_id: str, org_id: str, 
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get sessions by user ID."""
        cursor = self.collection.find({"user_id": user_id, "org_id": org_id}).sort("last_active", -1)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)
    
    async def update_last_active(self, session_id: str, org_id: str, user_id: Optional[str] = None) -> bool:
        """Update last active timestamp when this user owns the session."""
        query: Dict[str, Any] = {"session_id": session_id, "org_id": org_id}
        if user_id is not None:
            query["user_id"] = user_id
        result = await self.collection.update_one(
            query,
            {"$set": {"last_active": datetime.utcnow()}},
        )
        return result.matched_count > 0

    async def claim_patient(self, session_id: str, org_id: str, user_id: str, patient_id: str) -> bool:
        """Bind the first patient. A second patient does not overwrite the row."""
        result = await self.collection.update_one(
            {
                "session_id": session_id,
                "org_id": org_id,
                "user_id": user_id,
                "$or": [
                    {"patient_id": {"$in": [None, ""]}},
                    {"patient_id": {"$exists": False}},
                ],
            },
            {"$set": {"patient_id": patient_id}},
        )
        return result.matched_count > 0
    
    async def update_metadata(
        self,
        session_id: str,
        org_id: str,
        metadata: Dict[str, Any],
        user_id: str,
    ) -> bool:
        """Merge metadata keys when this user owns the session."""
        if not user_id or not metadata:
            return False
        if any(not isinstance(key, str) or "." in key or key.startswith("$") for key in metadata):
            return False
        result = await self.collection.update_one(
            {"session_id": session_id, "org_id": org_id, "user_id": user_id},
            {"$set": {f"metadata.{key}": value for key, value in metadata.items()}},
        )
        return result.matched_count > 0

    async def delete(self, session_id: str, org_id: str, user_id: str) -> bool:
        """Delete a session when this user owns it."""
        if not user_id:
            return False
        result = await self.collection.delete_one(
            {"session_id": session_id, "org_id": org_id, "user_id": user_id}
        )
        return result.deleted_count > 0
    
    async def count_by_org(self, org_id: str) -> int:
        """Count sessions for an organization."""
        return await self.collection.count_documents({"org_id": org_id})

