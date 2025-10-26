import pymongo
from pymongo import MongoClient
import json
from typing import List
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def store_chunks_to_mongodb(chunks_with_embeddings: List[dict], collection_name="chunks", batch_size=100):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medical_rag_db']
    coll = db[collection_name]
    
    success_count = 0
    failed_chunks = []
    
    for i in range(0, len(chunks_with_embeddings), batch_size):
        batch = chunks_with_embeddings[i:i+batch_size]
        
        # Filter out existing chunk_ids
        existing_ids = [doc['chunk_id'] for doc in coll.find({"chunk_id": {"$in": [chunk['chunk_id'] for chunk in batch]}}, {"chunk_id": 1})]
        new_batch = [chunk for chunk in batch if chunk['chunk_id'] not in existing_ids]
        
        if not new_batch:
            print(f"Batch {i//batch_size + 1} skipped: all chunks already exist")
            continue
        
        # Prepare new batch
        for chunk in new_batch:
            chunk['metadata']['created_timestamp'] = chunk['metadata']['created_timestamp'].replace(tzinfo=timezone.utc)
        
        try:
            result = coll.insert_many(new_batch, ordered=False)
            success_count += len(result.inserted_ids)
            print(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} chunks")
        except pymongo.errors.BulkWriteError as bwe:
            print(f"Batch error: {bwe.details}")
            failed_chunks.extend(new_batch)
        except Exception as e:
            print(f"Unexpected error: {e}")
            failed_chunks.extend(new_batch)
    
    if failed_chunks:
        with open('/fred/oz446/HenryNguyen/failed_chunks.json', 'w') as f:
            json.dump(failed_chunks, f, default=str)
        print(f"{len(failed_chunks)} failed; backed up")
    
    client.close()
    return success_count