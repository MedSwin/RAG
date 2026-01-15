"""Retrieval pipeline with two-stage retrieval, BM25, reranking, fusion, and MMR."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import tiktoken

from app.core.config import settings
from app.core.database import get_database
from app.core.indexing import (
    HNSWIndexBuilder,
    FAISSIndexBuilder,
    TreeIndexBuilder,
    load_hnsw_index,
    load_faiss_ivf_index,
    load_tree_index
)
from app.services.strategy import (
    IndexStrategyManager,
    IndexStrategy,
    analyze_query_characteristics
)
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient
from app.models.medswin import CandidatePassage, SourceType, EvidenceBundle
from app.repositories.chunks import ChunkRepository

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Two-stage retrieval pipeline with BM25, reranking, fusion, and MMR."""
    
    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
        chunk_repo: Optional[ChunkRepository] = None
    ):
        """Initialize retrieval pipeline.
        
        Args:
            embedding_client: Optional embedding client (uses endpoint if provided)
            reranker_client: Optional reranker client (uses endpoint if provided)
            chunk_repo: Optional chunk repository
        """
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.chunk_repo = chunk_repo or ChunkRepository()
        self.strategy_manager = IndexStrategyManager()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # For token counting
        
        # BM25 cache (per org_id)
        self._bm25_cache: Dict[str, BM25Okapi] = {}
    
    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        org_id: str,
        top_k: Optional[int] = None,
        source_type_filter: Optional[SourceType] = None,
        patient_id: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None
    ) -> List[CandidatePassage]:
        """Perform two-stage retrieval.
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            org_id: Organization ID
            top_k: Number of candidates to retrieve (defaults to CANDIDATE_K)
            source_type_filter: Optional source type filter
            patient_id: Optional patient ID filter
            hints: Optional retrieval hints (from sufficiency policy)
            
        Returns:
            List of candidate passages with dense and lexical scores
        """
        # Determine K based on hints or default
        k = top_k or settings.CANDIDATE_K
        if hints and hints.get("increase_k"):
            k = settings.CANDIDATE_K_PRIME
        
        # Stage 1: Dense retrieval
        dense_candidates = await self._dense_retrieval(
            query_embedding,
            org_id,
            k,
            source_type_filter,
            patient_id
        )
        
        # Stage 1: Lexical retrieval (BM25) if enabled
        lexical_candidates = []
        if settings.ENABLE_BM25:
            lexical_candidates = await self._lexical_retrieval(
                query,
                org_id,
                k,
                source_type_filter,
                patient_id
            )
        
        # Union candidates
        all_candidates = self._union_candidates(dense_candidates, lexical_candidates)
        
        # Normalize scores
        self._normalize_scores(all_candidates)
        
        return all_candidates[:k]  # Return top K
    
    async def _dense_retrieval(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str]
    ) -> List[CandidatePassage]:
        """Perform dense vector retrieval."""
        try:
            # Select index strategy
            query_char = analyze_query_characteristics(
                query="",  # Not used for dense retrieval
                top_k=k,
                use_reranking=True
            )
            strategy = self.strategy_manager.select_retrieval_strategy(query_char)
            
            # Load index (using existing infrastructure)
            embedding_dim = query_embedding.shape[0]
            index = await self._load_index(embedding_dim, strategy)
            
            # Query index
            labels, distances = index.query(query_embedding.reshape(1, -1), k)
            
            # Get chunks from database
            chunk_ids = []
            for label in labels[0]:
                chunk_id = index.mapping.get(str(int(label)))
                if chunk_id:
                    chunk_ids.append(chunk_id)
            
            # Fetch chunks with filters
            filter_dict = {"org_id": org_id, "chunk_id": {"$in": chunk_ids}}
            if source_type_filter:
                filter_dict["source_type"] = source_type_filter.value
            if patient_id:
                filter_dict["patient_id"] = patient_id
            
            db = get_database()
            chunks_cursor = db.chunks.find(filter_dict)
            chunks = await chunks_cursor.to_list(length=None)
            
            # Create candidate passages
            candidates = []
            chunk_dict = {c["chunk_id"]: c for c in chunks}
            distance_dict = {chunk_ids[i]: float(distances[0][i]) for i in range(len(chunk_ids))}
            
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                distance = distance_dict.get(chunk_id, 1.0)
                # Convert distance to similarity (1 - distance for cosine)
                dense_score = 1.0 - distance
                
                candidates.append(CandidatePassage(
                    chunk_id=chunk_id,
                    doc_id=chunk.get("doc_id", ""),
                    source_type=SourceType(chunk.get("source_type", "CPG")),
                    text=chunk.get("text", chunk.get("content", "")),
                    section=chunk.get("section"),
                    offset_start=chunk.get("offset_start"),
                    offset_end=chunk.get("offset_end"),
                    metadata=chunk.get("metadata", {}),
                    dense_score=dense_score
                ))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    async def _lexical_retrieval(
        self,
        query: str,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str]
    ) -> List[CandidatePassage]:
        """Perform BM25 lexical retrieval."""
        try:
            # Get or build BM25 index for org
            bm25 = await self._get_bm25_index(org_id, source_type_filter, patient_id)
            if not bm25:
                return []
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)
            
            # Get top K
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Fetch chunks
            db = get_database()
            filter_dict = {"org_id": org_id}
            if source_type_filter:
                filter_dict["source_type"] = source_type_filter.value
            if patient_id:
                filter_dict["patient_id"] = patient_id
            
            chunks_cursor = db.chunks.find(filter_dict)
            chunks = await chunks_cursor.to_list(length=None)
            
            # Create candidate passages
            candidates = []
            for idx in top_indices:
                if idx < len(chunks) and scores[idx] > 0:
                    chunk = chunks[idx]
                    candidates.append(CandidatePassage(
                        chunk_id=chunk["chunk_id"],
                        doc_id=chunk.get("doc_id", ""),
                        source_type=SourceType(chunk.get("source_type", "CPG")),
                        text=chunk.get("text", chunk.get("content", "")),
                        section=chunk.get("section"),
                        offset_start=chunk.get("offset_start"),
                        offset_end=chunk.get("offset_end"),
                        metadata=chunk.get("metadata", {}),
                        lexical_score=float(scores[idx])
                    ))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Lexical retrieval failed: {e}")
            return []
    
    async def _get_bm25_index(
        self,
        org_id: str,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str]
    ) -> Optional[BM25Okapi]:
        """Get or build BM25 index for organization."""
        cache_key = f"{org_id}_{source_type_filter}_{patient_id}"
        
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]
        
        try:
            # Fetch chunks
            db = get_database()
            filter_dict = {"org_id": org_id}
            if source_type_filter:
                filter_dict["source_type"] = source_type_filter.value
            if patient_id:
                filter_dict["patient_id"] = patient_id
            
            chunks_cursor = db.chunks.find(filter_dict, {"chunk_id": 1, "text": 1, "content": 1, "tokenized_text": 1})
            chunks = await chunks_cursor.to_list(length=None)
            
            if not chunks:
                return None
            
            # Tokenize texts
            tokenized_texts = []
            for chunk in chunks:
                if chunk.get("tokenized_text"):
                    tokenized_texts.append(chunk["tokenized_text"])
                else:
                    text = chunk.get("text") or chunk.get("content", "")
                    tokens = text.lower().split()
                    tokenized_texts.append(tokens)
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            self._bm25_cache[cache_key] = bm25
            
            return bm25
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return None
    
    def _union_candidates(
        self,
        dense_candidates: List[CandidatePassage],
        lexical_candidates: List[CandidatePassage]
    ) -> List[CandidatePassage]:
        """Union dense and lexical candidates."""
        # Create dict by chunk_id
        candidate_dict = {}
        
        for candidate in dense_candidates:
            candidate_dict[candidate.chunk_id] = candidate
        
        # Merge lexical scores
        for candidate in lexical_candidates:
            if candidate.chunk_id in candidate_dict:
                candidate_dict[candidate.chunk_id].lexical_score = candidate.lexical_score
            else:
                candidate_dict[candidate.chunk_id] = candidate
        
        return list(candidate_dict.values())
    
    def _normalize_scores(self, candidates: List[CandidatePassage]):
        """Normalize dense and lexical scores to [0, 1]."""
        dense_scores = [c.dense_score for c in candidates if c.dense_score is not None]
        lexical_scores = [c.lexical_score for c in candidates if c.lexical_score is not None]
        
        if dense_scores:
            min_dense = min(dense_scores)
            max_dense = max(dense_scores)
            if max_dense > min_dense:
                for candidate in candidates:
                    if candidate.dense_score is not None:
                        candidate.dense_score = (candidate.dense_score - min_dense) / (max_dense - min_dense)
        
        if lexical_scores:
            min_lex = min(lexical_scores)
            max_lex = max(lexical_scores)
            if max_lex > min_lex:
                for candidate in candidates:
                    if candidate.lexical_score is not None:
                        candidate.lexical_score = (candidate.lexical_score - min_lex) / (max_lex - min_lex)
    
    async def rerank(
        self,
        query: str,
        candidates: List[CandidatePassage]
    ) -> List[CandidatePassage]:
        """Rerank candidates using reranker service."""
        if not self.reranker_client or not candidates:
            return candidates
        
        try:
            # Extract passage texts
            passages = [c.text for c in candidates]
            
            # Call reranker
            rerank_results = await self.reranker_client.rerank(query, passages)
            
            # Update candidates with rerank scores
            for result in rerank_results:
                idx = result["index"]
                if idx < len(candidates):
                    candidates[idx].rerank_score = result.get("p_hat", result.get("score", 0.0))
                    candidates[idx].metadata["rerank_logit"] = result.get("logit")
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return candidates
    
    def compute_fusion_scores(self, candidates: List[CandidatePassage]) -> List[CandidatePassage]:
        """Compute fusion scores for candidates."""
        for candidate in candidates:
            fusion_score = 0.0
            
            # Reranker score
            if candidate.rerank_score is not None:
                fusion_score += settings.W_RERANK * candidate.rerank_score
            
            # Dense score
            if candidate.dense_score is not None:
                fusion_score += settings.W_DENSE * candidate.dense_score
            
            # Lexical score
            if candidate.lexical_score is not None:
                fusion_score += settings.W_LEX * candidate.lexical_score
            
            # Recency score (if timestamp available)
            recency_score = self._compute_recency_score(candidate)
            fusion_score += settings.W_RECENCY * recency_score
            
            # Section score
            section_score = self._compute_section_score(candidate)
            fusion_score += settings.W_SECTION * section_score
            
            # Source score
            source_score = self._compute_source_score(candidate)
            fusion_score += settings.W_SOURCE * source_score
            
            candidate.fusion_score = fusion_score
        
        # Sort by fusion score
        candidates.sort(key=lambda x: x.fusion_score or 0.0, reverse=True)
        
        return candidates
    
    def _compute_recency_score(self, candidate: CandidatePassage) -> float:
        """Compute recency score (0-1)."""
        # If timestamp in metadata, use it
        timestamp = candidate.metadata.get("timestamp")
        if timestamp:
            # Simple recency: newer = higher score
            # This is a placeholder - implement proper time-based scoring
            return 0.5  # Default
        return 0.5
    
    def _compute_section_score(self, candidate: CandidatePassage) -> float:
        """Compute section score (recommendations > background)."""
        section = candidate.section or ""
        section_lower = section.lower()
        
        if "recommendation" in section_lower or "guideline" in section_lower:
            return 1.0
        elif "background" in section_lower or "introduction" in section_lower:
            return 0.3
        else:
            return 0.7
    
    def _compute_source_score(self, candidate: CandidatePassage) -> float:
        """Compute source score (CPG vs EMR weighting)."""
        if candidate.source_type == SourceType.CPG:
            return 1.0
        elif candidate.source_type == SourceType.EMR:
            return 0.8
        else:
            return 0.6
    
    def select_with_mmr(
        self,
        candidates: List[CandidatePassage],
        query_embedding: np.ndarray,
        max_chunks: Optional[int] = None,
        token_budget: Optional[int] = None
    ) -> List[CandidatePassage]:
        """Select diverse passages using MMR under token budget."""
        if not candidates:
            return []
        
        max_chunks = max_chunks or settings.MMR_MAX_EVIDENCE_CHUNKS
        token_budget = token_budget or settings.TOKEN_BUDGET_B
        
        selected = []
        remaining = candidates.copy()
        
        # First, select highest scoring passage
        if remaining:
            selected.append(remaining.pop(0))
        
        # MMR selection
        while remaining and len(selected) < max_chunks:
            best_score = -float('inf')
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Relevance score (fusion score)
                relevance = candidate.fusion_score or 0.0
                
                # Diversity penalty (max similarity to already selected)
                max_sim = 0.0
                if selected:
                    # Compute similarity to selected passages
                    # For simplicity, use dense_score similarity
                    # In production, compute actual embedding similarity
                    for sel in selected:
                        # Placeholder: use fusion score difference as diversity proxy
                        sim = abs((candidate.fusion_score or 0.0) - (sel.fusion_score or 0.0))
                        max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = settings.MMR_LAMBDA * relevance - (1 - settings.MMR_LAMBDA) * max_sim
                
                # Check token budget
                tokens = len(self.tokenizer.encode(candidate.text))
                total_tokens = sum(len(self.tokenizer.encode(s.text)) for s in selected)
                
                if total_tokens + tokens <= token_budget:
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                # No more candidates fit in budget
                break
        
        return selected
    
    def build_evidence_bundle(
        self,
        passages: List[CandidatePassage]
    ) -> EvidenceBundle:
        """Build evidence bundle from selected passages."""
        total_tokens = sum(len(self.tokenizer.encode(p.text)) for p in passages)
        
        cpg_count = sum(1 for p in passages if p.source_type == SourceType.CPG)
        emr_count = sum(1 for p in passages if p.source_type == SourceType.EMR)
        lit_count = sum(1 for p in passages if p.source_type == SourceType.LIT)
        
        coverage_ratios = {
            "cpg_ratio": cpg_count / len(passages) if passages else 0.0,
            "emr_ratio": emr_count / len(passages) if passages else 0.0,
            "lit_ratio": lit_count / len(passages) if passages else 0.0
        }
        
        return EvidenceBundle(
            passages=passages,
            total_tokens=total_tokens,
            cpg_count=cpg_count,
            emr_count=emr_count,
            lit_count=lit_count,
            coverage_ratios=coverage_ratios
        )
    
    async def _load_index(
        self,
        embedding_dim: int,
        strategy: IndexStrategy
    ) -> Any:
        """Load index based on strategy."""
        if strategy == IndexStrategy.HNSW_ONLY:
            return load_hnsw_index(
                embedding_dim,
                settings.HNSW_INDEX_PATH,
                settings.HNSW_MAPPING_PATH
            )
        elif strategy == IndexStrategy.FAISS_ONLY:
            return load_faiss_ivf_index(
                embedding_dim,
                settings.FAISS_INDEX_PATH,
                settings.FAISS_MAPPING_PATH
            )
        elif strategy == IndexStrategy.TREE_ONLY:
            return load_tree_index(
                embedding_dim,
                settings.TREE_INDEX_PATH,
                settings.TREE_MAPPING_PATH
            )
        else:
            # Default to HNSW
            return load_hnsw_index(
                embedding_dim,
                settings.HNSW_INDEX_PATH,
                settings.HNSW_MAPPING_PATH
            )

