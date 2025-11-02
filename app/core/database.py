from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global database client
client: Optional[AsyncIOMotorClient] = None
sync_client: Optional[MongoClient] = None

async def init_database():
    """Initialize database connection."""
    global client, sync_client
    
    try:
        # Async client for FastAPI
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        
        # Sync client for existing code compatibility
        sync_client = MongoClient(settings.MONGODB_URL)
        
        # Test connection
        await client.admin.command('ping')
        logger.info("Database connection established")
        
        # Initialize collections and indexes
        await create_collections_and_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

async def create_collections_and_indexes():
    """Create collections and indexes."""
    if not client:
        raise Exception("Database client not initialized")
    
    db = client[settings.MONGODB_DATABASE]
    
    # Create chunks collection
    chunks_collection = db.chunks
    
    # Create indexes
    await chunks_collection.create_index("chunk_id", unique=True)
    await chunks_collection.create_index([("metadata.source", 1), ("metadata.task", 1)])
    await chunks_collection.create_index("metadata.parent_id")
    await chunks_collection.create_index("metadata.created_timestamp")
    await chunks_collection.create_index([("content", "text")])
    
    # Create chunk_relationships collection
    relationships_collection = db.chunk_relationships
    await relationships_collection.create_index("parent_chunk_id")
    await relationships_collection.create_index("relationship_type")
    
    # Create search_indexes collection
    search_indexes_collection = db.search_indexes
    await search_indexes_collection.create_index("index_type")
    await search_indexes_collection.create_index("last_updated")
    
    logger.info("Database collections and indexes created")

def get_database():
    """Get database instance."""
    if not client:
        raise Exception("Database client not initialized")
    return client[settings.MONGODB_DATABASE]

def get_sync_database():
    """Get synchronous database instance for compatibility."""
    if not sync_client:
        logger.warning("Sync database client not initialized - returning None")
        return None
    return sync_client[settings.MONGODB_DATABASE]

async def close_database():
    """Close database connection."""
    global client, sync_client
    
    if client:
        client.close()
        client = None
    
    if sync_client:
        sync_client.close()
        sync_client = None
    
    logger.info("Database connection closed")
