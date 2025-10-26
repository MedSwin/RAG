from pathlib import Path
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
import logging
from datetime import timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_embed_model(path: Path) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading embedding model on {device}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    embed_model = AutoModel.from_pretrained(path)
    embed_model = embed_model.to(device)
    embed_model.eval()
    
    # Detect embedding dimension by running a test input
    test_input = tokenizer("test", return_tensors="pt", truncation=True, padding=True, max_length=512)
    test_input = {k: v.to(device) for k, v in test_input.items()}
    with torch.no_grad():
        test_output = embed_model(**test_input)
        test_embedding = mean_pooling(test_output.last_hidden_state, test_input['attention_mask'])
        embedding_dim = test_embedding.shape[1]
    
    logger.info(f"Detected embedding dimension: {embedding_dim}")
    return tokenizer, embed_model, device, embedding_dim

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = summed_embeddings / summed_mask
    return mean_embeddings

def embedding_text(all_chunks: List[dict], tokenizer: PreTrainedTokenizer, embed_model: PreTrainedModel, device: torch.device, embedding_dim: int, batch_size: int = 64) -> tuple:
    all_embeddings = []
    valid_chunk_indices = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunk = all_chunks[i:i+batch_size]
        list_batch_chunk = []
        batch_indices = []
        
        for j, chunk in enumerate(batch_chunk):
            content = str(chunk["content"]).strip()
            if not content:
                logger.warning(f"Skipping chunk {chunk['metadata']['chunk_id']} due to empty content")
                continue
            list_batch_chunk.append(content)
            batch_indices.append(i + j)
            
        if not list_batch_chunk:
            logger.warning(f"Empty batch at index {i}, skipping")
            continue
        
        inputs = tokenizer(
            list_batch_chunk,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = embed_model(**inputs)
            
        attention_mask = inputs['attention_mask']
        chunk_embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
        
        for idx, emb in zip(batch_indices, chunk_embeddings):
            emb_np = emb.cpu().numpy().astype(np.float64)
            if len(emb_np) != embedding_dim or np.any(np.isnan(emb_np)) or np.any(np.isinf(emb_np)):
                logger.warning(f"Invalid embedding for chunk {all_chunks[idx]['metadata']['chunk_id']}: dim={len(emb_np)}, has NaN/Inf")
                continue
            all_embeddings.append(emb_np)
            valid_chunk_indices.append(idx)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    final_embeddings = torch.tensor(all_embeddings) if all_embeddings else torch.tensor([])
    logger.info(f"Generated embeddings shape: {final_embeddings.shape}")
    return final_embeddings, valid_chunk_indices

def add_embeddings_to_chunks(all_chunks: List[dict], final_embeddings: torch.Tensor, valid_indices: List[int], embedding_dim: int, embedded_model: str = "MedEmbed-large-v0.1") -> List[dict]:
    embeddings_with_chunks = []
    emb_idx = 0
    
    for i, chunk in enumerate(all_chunks):
        if i not in valid_indices:
            logger.warning(f"Skipping chunk {chunk['metadata']['chunk_id']} due to empty content or invalid embedding")
            continue
        
        embedding = final_embeddings[emb_idx].numpy().astype(np.float64)
        if len(embedding) != embedding_dim or np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.warning(f"Skipping chunk {chunk['metadata']['chunk_id']} due to invalid embedding: dim={len(embedding)}")
            emb_idx += 1
            continue
        
        meta = chunk["metadata"]
        embedded_chunk = {
            "chunk_id": str(meta.pop("chunk_id")),
            "content": str(chunk["content"]),
            "metadata": {
                "parent_id": str(meta["parent_id"]),
                "source": str(meta["source"]),
                "task": str(meta["task"]),
                "sequence": int(meta["sequence"]),
                "total_chunks": int(meta["total_chunks"]),
                "content_type": str(meta["content_type"]),
                "related_chunks": [str(c) for c in meta.get("related_chunks", [])],
                "chunk_length": int(meta["chunk_length"]),
                "created_timestamp": meta["created_timestamp"].replace(tzinfo=timezone.utc),
                "token_count": int(meta.get("token_count", 0))
            },
            "embedding": [float(x) for x in embedding.tolist()],
            "embedding_model": embedded_model,
            "embedding_dim": embedding_dim
        }
        embeddings_with_chunks.append(embedded_chunk)
        emb_idx += 1
    
    logger.info(f"Added embeddings to {len(embeddings_with_chunks)} chunks")
    return embeddings_with_chunks