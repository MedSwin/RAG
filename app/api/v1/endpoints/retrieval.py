from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import hnswlib
import json
import numpy as np
from pathlib import Path

from app.main import model_manager
from app.core.config import settings
from app.core.database import get_sync_database

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    """Request model for document retrieval."""
    query: str
    top_k: Optional[int] = None
    use_reranking: bool = True
    initial_top_k: Optional[int] = None
    final_top_k: Optional[int] = None

class DocumentResponse(BaseModel):
    """Response model for retrieved documents."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    rerank_score: Optional[float] = None
    # QCA format fields
    question: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None

class RetrievalResponse(BaseModel):
    """Response model for retrieval results."""
    query: str
    documents: List[DocumentResponse]
    total_documents: int
    retrieval_time: float
    used_reranking: bool

class IndexInfo(BaseModel):
    """Model for index information."""
    index_path: str
    mapping_path: str
    dimension: int
    total_vectors: int
    index_type: str

def get_embedding_model():
    """Dependency to get embedding model."""
    try:
        return model_manager.get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to get embedding model: {e}")
        raise HTTPException(status_code=503, detail="Embedding model not available")

def get_reranker_model():
    """Dependency to get reranker model."""
    return model_manager.get_reranker_model()

@router.post("/search", response_model=RetrievalResponse)
async def search_documents(
    request: QueryRequest,
    embedding_model = Depends(get_embedding_model),
    reranker_model = Depends(get_reranker_model)
):
    """Search for relevant documents."""
    import time
    start_time = time.time()
    
    try:
        # Set parameters
        top_k = request.top_k or settings.DEFAULT_TOP_K
        top_k = min(top_k, settings.MAX_TOP_K)
        
        initial_top_k = request.initial_top_k or settings.RERANK_TOP_K
        final_top_k = request.final_top_k or settings.FINAL_TOP_K
        
        # Get embedding model components
        tokenizer, embed_model, device, embedding_dim = embedding_model
        
        # Generate query embedding
        query_embedding = await _embed_query(
            request.query, 
            tokenizer, 
            embed_model, 
            device
        )
        
        # Load HNSW index
        index = await _load_hnsw_index(embedding_dim)
        
        # Retrieve documents
        if request.use_reranking and reranker_model:
            # Use reranking
            documents = await _retrieve_with_reranking(
                query_embedding,
                request.query,
                index,
                reranker_model,
                initial_top_k,
                final_top_k
            )
            used_reranking = True
        else:
            # Use simple retrieval
            documents = await _retrieve_simple(
                query_embedding,
                index,
                top_k
            )
            used_reranking = False
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResponse(
            query=request.query,
            documents=documents,
            total_documents=len(documents),
            retrieval_time=retrieval_time,
            used_reranking=used_reranking
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search", response_model=RetrievalResponse)
async def search_documents_get(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K),
    use_reranking: bool = Query(True, description="Use reranking"),
    initial_top_k: int = Query(settings.RERANK_TOP_K, ge=1, le=50),
    final_top_k: int = Query(settings.FINAL_TOP_K, ge=1, le=20),
    embedding_model = Depends(get_embedding_model),
    reranker_model = Depends(get_reranker_model)
):
    """Search for relevant documents using GET method."""
    request = QueryRequest(
        query=query,
        top_k=top_k,
        use_reranking=use_reranking,
        initial_top_k=initial_top_k,
        final_top_k=final_top_k
    )
    
    return await search_documents(request, embedding_model, reranker_model)

@router.get("/index/info", response_model=IndexInfo)
async def get_index_info():
    """Get information about the HNSW index."""
    try:
        index_path = Path(settings.HNSW_INDEX_PATH)
        mapping_path = Path(settings.HNSW_MAPPING_PATH)
        
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="HNSW index not found")
        
        if not mapping_path.exists():
            raise HTTPException(status_code=404, detail="HNSW mapping not found")
        
        # Load mapping to get total vectors
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        # Get embedding dimension from model
        try:
            _, _, _, embedding_dim = model_manager.get_embedding_model()
        except:
            embedding_dim = settings.EMBEDDING_DIMENSION
        
        return IndexInfo(
            index_path=str(index_path),
            mapping_path=str(mapping_path),
            dimension=embedding_dim,
            total_vectors=len(mapping),
            index_type="HNSW"
        )
        
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index info: {str(e)}")

async def _embed_query(query: str, tokenizer, embed_model, device) -> np.ndarray:
    """Embed a query string."""
    import torch
    import torch.nn.functional as F
    
    inputs = tokenizer(
        [query],
        truncation=True,
        padding=True,
        max_length=settings.MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embed_model(**inputs)
        attention_mask = inputs['attention_mask']
        query_embedding = _mean_pooling(outputs.last_hidden_state, attention_mask)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    return query_embedding.cpu().numpy().astype(np.float64)

async def _load_hnsw_index(embedding_dim: int) -> hnswlib.Index:
    """Load HNSW index."""
    index_path = Path(settings.HNSW_INDEX_PATH)
    
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="HNSW index not found")
    
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.load_index(str(index_path))
    
    return index

async def _retrieve_simple(
    query_embedding: np.ndarray,
    index: hnswlib.Index,
    top_k: int
) -> List[DocumentResponse]:
    """Simple retrieval without reranking."""
    # Ensure query embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Search index
    labels, distances = index.knn_query(query_embedding, k=top_k)
    
    # Load mapping
    mapping_path = Path(settings.HNSW_MAPPING_PATH)
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Get database
    db = get_sync_database()
    coll = db['chunks']
    
    # Retrieve documents
    documents = []
    for label, distance in zip(labels[0], distances[0]):
        chunk_id = mapping.get(str(int(label)))
        if not chunk_id:
            continue
        
        chunk = coll.find_one(
            {"chunk_id": chunk_id}, 
            {"chunk_id": 1, "content": 1, "metadata": 1}
        )
        
        if chunk:
            # Extract QCA format from chunk content
            qca_data = _extract_qca_from_chunk(chunk)
            
            documents.append(DocumentResponse(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                distance=float(distance),
                question=qca_data.get("question"),
                context=qca_data.get("context"),
                answer=qca_data.get("answer")
            ))
    
    return documents

async def _retrieve_with_reranking(
    query_embedding: np.ndarray,
    query: str,
    index: hnswlib.Index,
    reranker_model,
    initial_top_k: int,
    final_top_k: int
) -> List[DocumentResponse]:
    """Retrieve documents with reranking."""
    # First, retrieve more documents
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    labels, distances = index.knn_query(query_embedding, k=initial_top_k)
    
    # Load mapping
    mapping_path = Path(settings.HNSW_MAPPING_PATH)
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Get database
    db = get_sync_database()
    coll = db['chunks']
    
    # Retrieve initial documents
    initial_docs = []
    for label, distance in zip(labels[0], distances[0]):
        chunk_id = mapping.get(str(int(label)))
        if not chunk_id:
            continue
        
        chunk = coll.find_one(
            {"chunk_id": chunk_id}, 
            {"chunk_id": 1, "content": 1, "metadata": 1}
        )
        
        if chunk:
            initial_docs.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "distance": float(distance)
            })
    
    # Rerank documents
    reranked_docs = reranker_model.rerank_documents(
        query=query,
        documents=initial_docs,
        top_k=final_top_k
    )
    
    # Convert to response format
    documents = []
    for doc in reranked_docs:
        # Extract QCA format from chunk content
        qca_data = _extract_qca_from_chunk(doc)
        
        documents.append(DocumentResponse(
            chunk_id=doc["chunk_id"],
            content=doc["content"],
            metadata=doc["metadata"],
            distance=doc["distance"],
            rerank_score=doc.get("rerank_score"),
            question=qca_data.get("question"),
            context=qca_data.get("context"),
            answer=qca_data.get("answer")
        ))
    
    return documents

def _extract_qca_from_chunk(chunk: Dict[str, Any]) -> Dict[str, str]:
    """Extract Question, Context, Answer from chunk content."""
    try:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        # Try to extract from metadata first (if stored separately)
        if "question" in metadata and "context" in metadata and "answer" in metadata:
            return {
                "question": metadata["question"],
                "context": metadata["context"],
                "answer": metadata["answer"]
            }
        
        # Parse from content using patterns
        import re
        
        # Look for QCA patterns in content
        question_pattern = r"(?:Question|Patient Question|Input):\s*(.*?)(?=(?:Context|Answer|Output|Doctor Response):|$)"
        context_pattern = r"(?:Context|Patient Question \(continued\)):\s*(.*?)(?=(?:Answer|Output|Doctor Response):|$)"
        answer_pattern = r"(?:Answer|Output|Doctor Response|Doctor Response \(continued\)):\s*(.*?)$"
        
        question_match = re.search(question_pattern, content, re.DOTALL | re.IGNORECASE)
        context_match = re.search(context_pattern, content, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(answer_pattern, content, re.DOTALL | re.IGNORECASE)
        
        question = question_match.group(1).strip() if question_match else ""
        context = context_match.group(1).strip() if context_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # If no patterns found, try to split by common separators
        if not question and not context and not answer:
            parts = content.split("\n\n")
            if len(parts) >= 2:
                question = parts[0].strip()
                answer = parts[-1].strip()
                context = "\n\n".join(parts[1:-1]).strip() if len(parts) > 2 else ""
        
        return {
            "question": question or "No question found",
            "context": context or "No additional context",
            "answer": answer or "No answer found"
        }
        
    except Exception as e:
        logger.warning(f"Error extracting QCA from chunk: {e}")
        return {
            "question": "Error extracting question",
            "context": "Error extracting context", 
            "answer": "Error extracting answer"
        }

def _mean_pooling(token_embeddings, attention_mask):
    """Mean pooling function."""
    import torch
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * input_mask_expanded
    summed_embeddings = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = summed_embeddings / summed_mask
    return mean_embeddings
