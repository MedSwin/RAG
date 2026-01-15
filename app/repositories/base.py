"""Base repository class with org-aware partitioning."""

from typing import Optional, Dict, Any, List
from app.core.database import get_database
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with org-aware partitioning."""
    
    def __init__(self, collection_name: str):
        """Initialize repository with collection name."""
        self.collection_name = collection_name
        self._db: Optional[AsyncIOMotorDatabase] = None
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self._db is None:
            self._db = get_database()
        return self._db
    
    @property
    def collection(self):
        """Get collection with org-aware partitioning."""
        return self.db[self.collection_name]
    
    def _ensure_org_id(self, data: Dict[str, Any], org_id: str) -> Dict[str, Any]:
        """Ensure org_id is present in data."""
        if "org_id" not in data:
            data["org_id"] = org_id
        return data
    
    async def create_indexes(self):
        """Create indexes for the collection. Override in subclasses."""
        pass

