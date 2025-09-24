import hnswlib
import numpy as np
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_embed_model(path: Path) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading embedding model on {device}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    embed_model = AutoModel.from_pretrained(path)
    embed_model = embed_model.to(device)
    embed_model.eval()
    return tokenizer, embed_model, device

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = summed_embeddings / summed_mask
    return mean_embeddings

def embed_query(query: str, tokenizer, embed_model, device) -> np.ndarray:
    inputs = tokenizer(
        [query],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    attention_mask = inputs['attention_mask']
    query_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().numpy().astype(np.float64)

def retrieve_chunks(query: str, model_path: str, index_path: str, mapping_path: str = "/fred/oz446/HenryNguyen/data/hnsw_mapping.json", collection_name: str = "chunks", top_k: int = 5) -> List[dict]:
    tokenizer, embed_model, device = load_embed_model(Path(model_path))
    
    logger.info(f"Embedding query: {query}")
    query_embedding = embed_query(query, tokenizer, embed_model, device)
    if query_embedding.shape[1] != 768:
        logger.error(f"Invalid query embedding dimension: {query_embedding.shape[1]}")
        return []
    
    logger.info(f"Loading HNSW index from {index_path}")
    index = hnswlib.Index(space='cosine', dim=768)
    index.load_index(index_path)
    
    labels, distances = index.knn_query(query_embedding, k=top_k)
    logger.info(f"Retrieved {len(labels[0])} chunks with distances: {distances[0]}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medical_rag_db']
    coll = db[collection_name]
    
    results = []
    for label, distance in zip(labels[0], distances[0]):
        chunk_id = mapping.get(str(int(label)))
        if not chunk_id:
            logger.warning(f"No mapping for label {label}")
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
            logger.warning(f"Chunk with chunk_id {chunk_id} not found in MongoDB")
    
    client.close()
    return results

if __name__ == "__main__":
    query = "What are the symptoms of diabetes?"
    model_path = "/fred/oz446/HenryNguyen/EmbeddingModel/PubMedBERT-MNLI-MedNLI"
    index_path = "/fred/oz446/HenryNguyen/data/hnsw_index.bin"
    
    results = retrieve_chunks(query, model_path, index_path, top_k=5)
    
    for i, result in enumerate(results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"Chunk ID: {result['chunk_id']}")
        logger.info(f"Content: {result['content'][:200]}...")
        logger.info(f"Distance: {result['distance']}")
        logger.info(f"Metadata: {result['metadata']}")
        logger.info("-" * 50)