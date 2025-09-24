import hnswlib
import numpy as np
from pymongo import MongoClient
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_hnsw_index(collection_name="chunks", index_path="/fred/oz446/HenryNguyen/data/hnsw_index.bin", mapping_path="/fred/oz446/HenryNguyen/data/hnsw_mapping.json"):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medical_rag_db']
    coll = db[collection_name]
    
    data = list(coll.find({}, {'chunk_id': 1, 'embedding': 1}))
    embeddings = np.array([d['embedding'] for d in data], dtype=np.float32)
    chunk_ids = [d['chunk_id'] for d in data]
    
    # Use integer labels (0 to n-1)
    labels = list(range(len(data)))
    
    index = hnswlib.Index(space='cosine', dim=768)
    index.init_index(max_elements=len(data), ef_construction=200, M=16)
    index.add_items(embeddings, labels)
    index.save_index(index_path)
    
    # Save mapping: integer label to chunk_id
    mapping = {int(label): chunk_id for label, chunk_id in zip(labels, chunk_ids)}
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f)
    
    print(f"HNSW index saved to {index_path}, mapping to {mapping_path}")
    
    client.close()