try:
    from motor.motor_asyncio import AsyncIOMotorClient
except (ModuleNotFoundError, ImportError):
    # Root Cause vs Logic: Motor imports a private PyMongo cursor symbol that
    # changed across driver releases, so a mismatched local install can crash
    # application import before startup reaches the graceful DB fallback path.
    AsyncIOMotorClient = None

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None
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
        if AsyncIOMotorClient is None or MongoClient is None:
            raise RuntimeError("MongoDB dependencies are not installed")
        # Async client for FastAPI
        client = AsyncIOMotorClient(settings.MONGODB_URL)

        # Sync client for existing code compatibility
        sync_client = MongoClient(settings.MONGODB_URL)

        # Test connection
        await client.admin.command("ping")
        sync_client.admin.command("ping")
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
    await chunks_collection.create_index([("org_id", 1), ("chunk_id", 1)])
    await chunks_collection.create_index([("org_id", 1), ("source_type", 1)])
    await chunks_collection.create_index([("org_id", 1), ("patient_id", 1)])
    await chunks_collection.create_index([("org_id", 1), ("doc_id", 1)])
    await chunks_collection.create_index([("metadata.source", 1), ("metadata.task", 1)])
    await chunks_collection.create_index("metadata.parent_id")
    await chunks_collection.create_index("metadata.created_timestamp")
    await chunks_collection.create_index([("content", "text")])

    # Motivation vs Logic: MedSwin traces and EMR/CPG documents must be
    # replayable by tenant and patient scope. These indexes make policy audits
    # and scoped retrieval first-class instead of best-effort collection scans.
    documents_collection = db.documents
    await documents_collection.create_index("doc_id", unique=True)
    await documents_collection.create_index([("org_id", 1), ("doc_id", 1)])
    await documents_collection.create_index([("org_id", 1), ("source_type", 1)])
    await documents_collection.create_index([("org_id", 1), ("patient_id", 1)])
    await documents_collection.create_index([("org_id", 1), ("effective_date", -1)])

    sessions_collection = db.sessions
    await sessions_collection.create_index("session_id", unique=True)
    await sessions_collection.create_index([("org_id", 1), ("session_id", 1)])
    await sessions_collection.create_index([("org_id", 1), ("user_id", 1)])

    traces_collection = db.traces
    await traces_collection.create_index("trace_id", unique=True)
    await traces_collection.create_index([("org_id", 1), ("trace_id", 1)])
    await traces_collection.create_index([("org_id", 1), ("session_id", 1)])
    await traces_collection.create_index([("org_id", 1), ("patient_id", 1)])

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
    """Get asynchronous database instance."""
    if not client:
        raise Exception("Database client not initialized")
    return client[settings.MONGODB_DATABASE]


def get_sync_database():
    """Get a synchronous database instance, initializing it when necessary.

    CLI/evaluation scripts run outside FastAPI's lifespan, so they cannot rely
    on ``init_database()`` having populated the global sync client first. A lazy
    connection keeps those scripts executable while preserving the shared client
    for normal application startup.
    """
    global sync_client
    if sync_client is None:
        if MongoClient is None:
            raise RuntimeError("PyMongo is not installed")
        sync_client = MongoClient(settings.MONGODB_URL)
        sync_client.admin.command("ping")
        logger.info("Lazy synchronous database connection established")
    return sync_client[settings.MONGODB_DATABASE]


async def close_database():
    """Close database connections."""
    global client, sync_client

    if client:
        client.close()
        client = None
    if sync_client:
        sync_client.close()
        sync_client = None
    logger.info("Database connection closed")
