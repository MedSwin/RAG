import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["medical_rag_db"]

db.drop_collection("chunks")
db.drop_collection("chunk_relationships")
db.drop_collection("search_indexes")

chunks_collection = db.create_collection("chunks")
chunks_collection.create_index("chunk_id", unique=True)
chunks_collection.create_index([("metadata.source", 1), ("metadata.task", 1)])
chunks_collection.create_index("metadata.parent_id")
chunks_collection.create_index("metadata.created_timestamp")

# Validator for "chunk" collection
db.command({
    "collMod": "chunks",
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            'required': ['chunk_id', 'content', 'embedding', 'metadata'],
            'properties': {
                'chunk_id': {'bsonType': 'string'},
                'content': {'bsonType': 'string'},
                'embedding': {'bsonType': 'array', 'items': {'bsonType': 'double'}},
                'metadata': {
                    'bsonType': 'object',
                    'required': ['parent_id', 'source', 'task', 'sequence', 'total_chunks', 'content_type'],
                    'properties': {
                        'parent_id': {'bsonType': 'string'},
                        'source': {'bsonType': 'string'},
                        'task': {'bsonType': 'string'},
                        'sequence': {'bsonType': 'int'},
                        'total_chunks': {'bsonType': 'int'},
                        'content_type': {'bsonType': 'string'},
                        'related_chunks': {'bsonType': 'array', 'items': {'bsonType': 'string'}},
                        'chunk_length': {'bsonType': 'int'},
                        'created_timestamp': {'bsonType': 'date'}
                    }
                }
            }
        }
    },
    "validationLevel": 'moderate'
})

#optimize the indexes
chunks_collection.create_index(
    [
        ("metadata.content_type", 1),
        ("metadata.source", 1),
        ("metadata.task", 1),
        ("metadata.parent_id", 1),
        ("metadata.created_timestamp", 1),
    ],
    name="metadata_search"
)

chunks_collection.create_index(
    [("content", "text")],
    name="text_search"
)


# chunk_rellationships 
chunk_rellationships_collection = db.create_collection("chunk_relationships")
chunk_rellationships_collection.create_index('parent_chunk_id')
chunk_rellationships_collection.create_index('relationship_type')
db.command({
    "collMod": "chunk_relationships",
    'validator': {
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['parent_chunk_id', 'child_chunk_ids', 'relationship_type'],
            'properties': {
                'parent_chunk_id': {'bsonType': 'string'},
                'child_chunk_ids': {'bsonType': 'array', 'items': {'bsonType': 'string'}},
                'relationship_type': {'enum': ['split_from', 'related_to', 'follows']},
                'strength_score': {'bsonType': 'double', 'minimum': 0.0, 'maximum': 1.0}
            }
        }
    }
})

#search_indexes
search_idx_collection = db.create_collection("search_indexes")
search_idx_collection.create_index("index_type")
search_idx_collection.create_index("last_updated")
db.command({
    "collMod": "search_indexes",
    "validator": {
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['index_type', 'chunk_ids', 'index_metadata', 'last_updated'],
            'properties': {
                'index_type': {'enum': ['summary', 'medical_condition', 'treatment']},
                'chunk_ids': {'bsonType': 'array', 'items': {'bsonType': 'string'}},
                'index_metadata': {'bsonType': 'object'},
                'last_updated': {'bsonType': 'date'}
            }
        }
    }
})

print("Create the schema successfully!!!")

client.close()