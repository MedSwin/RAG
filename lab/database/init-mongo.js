// MongoDB initialization script
db = db.getSiblingDB('medical_rag_db');

// Create collections
db.createCollection('chunks');
db.createCollection('chunk_relationships');
db.createCollection('search_indexes');

// Create indexes for chunks collection
db.chunks.createIndex({ "chunk_id": 1 }, { unique: true });
db.chunks.createIndex({ "metadata.source": 1, "metadata.task": 1 });
db.chunks.createIndex({ "metadata.parent_id": 1 });
db.chunks.createIndex({ "metadata.created_timestamp": 1 });
db.chunks.createIndex({ "content": "text" });

// Create indexes for chunk_relationships collection
db.chunk_relationships.createIndex({ "parent_chunk_id": 1 });
db.chunk_relationships.createIndex({ "relationship_type": 1 });

// Create indexes for search_indexes collection
db.search_indexes.createIndex({ "index_type": 1 });
db.search_indexes.createIndex({ "last_updated": 1 });

// Create validation schemas
db.runCommand({
    "collMod": "chunks",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["chunk_id", "content", "embedding", "metadata", "embedding_model", "embedding_dim"],
            "properties": {
                "chunk_id": { "bsonType": "string" },
                "content": { "bsonType": "string" },
                "embedding": { "bsonType": "array", "items": { "bsonType": "double" } },
                "embedding_model": { "bsonType": "string" },
                "embedding_dim": { "bsonType": "int", "minimum": 1 },
                "metadata": {
                    "bsonType": "object",
                    "required": ["parent_id", "source", "task", "sequence", "total_chunks", "content_type"],
                    "properties": {
                        "parent_id": { "bsonType": "string" },
                        "source": { "bsonType": "string" },
                        "task": { "bsonType": "string" },
                        "sequence": { "bsonType": "int" },
                        "total_chunks": { "bsonType": "int" },
                        "content_type": { "bsonType": "string" },
                        "related_chunks": { "bsonType": "array", "items": { "bsonType": "string" } },
                        "chunk_length": { "bsonType": "int" },
                        "created_timestamp": { "bsonType": "date" }
                    }
                }
            }
        }
    },
    "validationLevel": "moderate"
});

db.runCommand({
    "collMod": "chunk_relationships",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["parent_chunk_id", "child_chunk_ids", "relationship_type"],
            "properties": {
                "parent_chunk_id": { "bsonType": "string" },
                "child_chunk_ids": { "bsonType": "array", "items": { "bsonType": "string" } },
                "relationship_type": { "enum": ["split_from", "related_to", "follows"] },
                "strength_score": { "bsonType": "double", "minimum": 0.0, "maximum": 1.0 }
            }
        }
    }
});

db.runCommand({
    "collMod": "search_indexes",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["index_type", "chunk_ids", "index_metadata", "last_updated"],
            "properties": {
                "index_type": { "enum": ["summary", "medical_condition", "treatment"] },
                "chunk_ids": { "bsonType": "array", "items": { "bsonType": "string" } },
                "index_metadata": { "bsonType": "object" },
                "last_updated": { "bsonType": "date" }
            }
        }
    }
});

print("Database initialization completed successfully!");
