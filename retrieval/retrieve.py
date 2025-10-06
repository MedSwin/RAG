
from typing import List
import hnswlib
import json
from pymongo import MongoClient

def retrieve_chunks(query_embedding, index_path: str, collection_name:str = "chunks", mapping_path:str = "/fred/oz446/HenryNguyen/data/hnsw_mapping.json", top_k: int = 5) -> List[dict]:
    
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Get embedding dimension from MongoDB to validate query
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medical_rag_db']
    coll = db[collection_name]
    
    # Get embedding dimension from first document
    sample_doc = coll.find_one({}, {'embedding_dim': 1})
    if not sample_doc:
        print("No documents found in collection")
        client.close()
        return []
    
    expected_dim = sample_doc.get('embedding_dim', 768)
    
    if query_embedding.shape[1] != expected_dim:
        print(f"Invalid query embedding dimension: got {query_embedding.shape[1]}, expected {expected_dim}")
        client.close()
        return []
    
    index = hnswlib.Index(space='cosine', dim=expected_dim)
    index.load_index(index_path)
    
    labels, distances = index.knn_query(query_embedding, k=top_k)
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    results = []
    for label, distance in zip(labels[0], distances[0]):
        chunk_id = mapping.get(str(int(label)))
        if not chunk_id:
            print(f"No mapping for label {label}")
            continue
        chunk = coll.find_one({"chunk_id": chunk_id}, {"chunk_id": 1, "content": 1, "metadata": 1})
        if chunk:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "distance": float(distance)
            })
        else:
            print(f"Chunk with chunk_id {chunk_id} not found in MongoDB")
    
    client.close()
    return results