try:
    from motor.motor_asyncio import AsyncIOMotorClient
except (ModuleNotFoundError, ImportError):
    AsyncIOMotorClient = None

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

client: Optional[AsyncIOMotorClient] = None
sync_client: Optional[MongoClient] = None


async def init_database():
    """Initialize database connection and tenant-safe persistence indexes."""
    global client, sync_client

    try:
        if AsyncIOMotorClient is None or MongoClient is None:
            raise RuntimeError("MongoDB dependencies are not installed")
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        sync_client = MongoClient(settings.MONGODB_URL)
        await client.admin.command("ping")
        sync_client.admin.command("ping")
        logger.info("Database connection established")
        await create_collections_and_indexes()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def _drop_legacy_chunk_indexes(collection) -> None:
    """Migrate historical global identity and Mongo text indexes.

    Chunk lexical retrieval is owned by MedSwin BM25/SQLite FTS, not Mongo
    ``$text``. Maintaining a second full-body text index duplicates the corpus
    and is especially costly for the complete TREC build.
    """
    info = await collection.index_information()
    for name, spec in info.items():
        keys = spec.get("key") or []
        if keys == [("chunk_id", 1)] and spec.get("unique"):
            await collection.drop_index(name)
            logger.info("Dropped legacy global unique chunk_id index %s", name)
            continue
        if any(str(kind) == "text" for _field, kind in keys):
            await collection.drop_index(name)
            logger.info("Dropped redundant Mongo chunk text index %s", name)


async def _drop_legacy_document_indexes(collection) -> None:
    """Remove historical global document identity before compound uniqueness."""
    info = await collection.index_information()
    for name, spec in info.items():
        keys = spec.get("key") or []
        if keys == [("doc_id", 1)] and spec.get("unique"):
            await collection.drop_index(name)
            logger.info("Dropped legacy global unique doc_id index %s", name)


async def create_collections_and_indexes():
    """Create collections and indexes matching the repository tenancy contract."""
    if not client:
        raise Exception("Database client not initialized")

    db = client[settings.MONGODB_DATABASE]

    chunks_collection = db.chunks
    await _drop_legacy_chunk_indexes(chunks_collection)
    await chunks_collection.create_index([("org_id", 1), ("chunk_id", 1)], unique=True)
    await chunks_collection.create_index([("org_id", 1), ("source_type", 1)])
    await chunks_collection.create_index([("org_id", 1), ("patient_id", 1)])
    await chunks_collection.create_index([("org_id", 1), ("doc_id", 1)])
    await chunks_collection.create_index([("metadata.source", 1), ("metadata.task", 1)])
    await chunks_collection.create_index("metadata.parent_id")
    await chunks_collection.create_index("metadata.created_timestamp")

    documents_collection = db.documents
    await _drop_legacy_document_indexes(documents_collection)
    await documents_collection.create_index([("org_id", 1), ("doc_id", 1)], unique=True)
    await documents_collection.create_index([("org_id", 1), ("source_type", 1)])
    await documents_collection.create_index([("org_id", 1), ("patient_id", 1)])
    await documents_collection.create_index([("org_id", 1), ("effective_date", -1)])

    # Session/trace IDs are generated UUID identities and intentionally remain
    # globally unique, while compound org indexes support scoped reads.
    sessions_collection = db.sessions
    await sessions_collection.create_index("session_id", unique=True)
    await sessions_collection.create_index([("org_id", 1), ("session_id", 1)])
    await sessions_collection.create_index([("org_id", 1), ("user_id", 1)])

    traces_collection = db.traces
    await traces_collection.create_index("trace_id", unique=True)
    await traces_collection.create_index([("org_id", 1), ("trace_id", 1)])
    await traces_collection.create_index([("org_id", 1), ("session_id", 1)])
    await traces_collection.create_index([("org_id", 1), ("patient_id", 1)])

    relationships_collection = db.chunk_relationships
    await relationships_collection.create_index("parent_chunk_id")
    await relationships_collection.create_index("relationship_type")

    search_indexes_collection = db.search_indexes
    await search_indexes_collection.create_index("index_type")
    await search_indexes_collection.create_index("last_updated")

    logger.info("Database collections and tenant-safe indexes created")


def get_database():
    """Get asynchronous database instance."""
    if not client:
        raise Exception("Database client not initialized")
    return client[settings.MONGODB_DATABASE]


def get_sync_database():
    """Get a synchronous database instance, initializing it when necessary.

    CLI/evaluation scripts run outside FastAPI's lifespan, so they cannot rely
    on ``init_database()`` having populated the global sync client first.
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
