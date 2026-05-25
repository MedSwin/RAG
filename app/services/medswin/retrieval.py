"""Retrieval pipeline with two-stage retrieval, BM25, reranking, fusion, and MMR."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import math
import numpy as np

try:
    import tiktoken
except ModuleNotFoundError:
    class _WhitespaceEncoding:
        def encode(self, text):
            return text.split()

    class tiktoken:
        @staticmethod
        def get_encoding(_name):
            return _WhitespaceEncoding()

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    class BM25Okapi:
        """Small fallback scorer used when rank-bm25 is unavailable."""

        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query_tokens):
            query = set(query_tokens)
            scores = []
            for doc in self.corpus:
                doc_terms = set(doc)
                scores.append(float(len(query & doc_terms)) / max(len(query), 1))
            return np.array(scores, dtype=np.float32)

from app.core.config import settings
from app.core.database import get_database
try:
    from app.core.indexing import (
        HNSWIndexBuilder,
        FAISSIndexBuilder,
        TreeIndexBuilder,
        load_hnsw_index,
        load_faiss_ivf_index,
        load_tree_index
    )
except ModuleNotFoundError:
    HNSWIndexBuilder = FAISSIndexBuilder = TreeIndexBuilder = None

    def _missing_index_loader(*_args, **_kwargs):
        raise RuntimeError("Vector index dependencies are not installed")

    load_hnsw_index = load_faiss_ivf_index = load_tree_index = _missing_index_loader
from app.services.strategy import (
    IndexStrategyManager,
    IndexStrategy,
    analyze_query_characteristics
)
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient
from app.services.storage import StorageService
from app.models.medswin import (
    CandidatePassage,
    SourceType,
    EvidenceBundle,
    ClinicalFacet,
    EvidenceLedgerEntry,
    FacetCoverage,
    ContradictionPair,
    PolicyDecision,
)
from app.repositories.chunks import ChunkRepository
from app.services.medswin.governance import clamp, evidence_grade_from_metadata

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
        self._bm25_cache: Dict[str, Dict[str, Any]] = {}
    
    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        org_id: str,
        top_k: Optional[int] = None,
        source_type_filter: Optional[SourceType] = None,
        patient_id: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
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
            patient_id,
            constraints
        )
        
        # Stage 1: Lexical retrieval (BM25) if enabled
        lexical_candidates = []
        if settings.ENABLE_BM25:
            lexical_candidates = await self._lexical_retrieval(
                query,
                org_id,
                k,
                source_type_filter,
                patient_id,
                constraints
            )
        
        # Union candidates
        all_candidates = self._union_candidates(dense_candidates, lexical_candidates)

        # Motivation vs Logic: clinician CDS requests often need both patient
        # context and literature. A single mixed-source ANN query can be
        # dominated by the patient note and starve the LIT facet, so ANY-source
        # runs add bounded per-source probes before reranking/selection.
        if self._should_source_balance(source_type_filter, constraints):
            balanced_candidates = await self._source_balanced_retrieval(
                query=query,
                query_embedding=query_embedding,
                org_id=org_id,
                k=max(1, k // 2),
                patient_id=patient_id,
                hints=hints,
                constraints=constraints,
            )
            all_candidates = self._union_candidates(all_candidates, balanced_candidates)
        
        # Normalize scores
        self._normalize_scores(all_candidates)

        # Root Cause vs Logic: the old code sliced the merged candidate pool
        # before reranking based on an arbitrary list order, which could discard
        # high-value literature chunks and keep only the first few dense hits.
        # Logic: return the full retrieved pool so later reranking and MMR can
        # choose evidence by score rather than insertion order.
        all_candidates.sort(
            key=lambda c: (
                c.dense_score or 0.0,
                c.lexical_score or 0.0,
            ),
            reverse=True,
        )
        return all_candidates

    def _should_source_balance(
        self,
        source_type_filter: Optional[SourceType],
        constraints: Optional[Dict[str, Any]],
    ) -> bool:
        if source_type_filter is not None:
            return False
        constraints = constraints or {}
        return str(constraints.get("source_policy") or "ANY").upper() == "ANY"

    async def _source_balanced_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        patient_id: Optional[str],
        hints: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
    ) -> List[CandidatePassage]:
        balanced: List[CandidatePassage] = []
        for source_type in (SourceType.LIT, SourceType.EMR):
            source_constraints = dict(constraints or {})
            source_constraints["source_policy"] = f"{source_type.value}_ONLY"
            dense = await self._dense_retrieval(query_embedding, org_id, k, source_type, patient_id, source_constraints)
            lexical = []
            if settings.ENABLE_BM25:
                lexical = await self._lexical_retrieval(query, org_id, k, source_type, patient_id, source_constraints)
            balanced.extend(self._union_candidates(dense, lexical))
        return balanced
    
    async def _dense_retrieval(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]] = None
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
            labels, distances = await self._query_index_with_rebuild(index, query_embedding, k, embedding_dim, strategy)
            labels = np.asarray(labels).reshape(-1)
            distances = np.asarray(distances).reshape(-1)
            
            # Get chunks from database
            chunk_ids = []
            for label in labels:
                chunk_id = index.mapping.get(str(int(label)))
                if chunk_id:
                    chunk_ids.append(chunk_id)
            
            # Fetch chunks with filters
            filter_dict = self._retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            filter_dict["chunk_id"] = {"$in": chunk_ids}
            
            db = get_database()
            chunks_cursor = db.chunks.find(filter_dict)
            chunks = await chunks_cursor.to_list(length=None)
            
            # Create candidate passages
            candidates = []
            chunk_dict = {c["chunk_id"]: c for c in chunks}
            distance_dict = {
                chunk_ids[i]: float(distances[i])
                for i in range(min(len(chunk_ids), len(distances)))
            }
            
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
                    token_count=chunk.get("token_count") or chunk.get("metadata", {}).get("token_count"),
                    evidence_grade_score=(chunk.get("evidence_grade") or {}).get("score") if isinstance(chunk.get("evidence_grade"), dict) else None,
                    dense_score=dense_score
                ))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    async def _query_index_with_rebuild(
        self,
        index: Any,
        query_embedding: np.ndarray,
        k: int,
        embedding_dim: int,
        strategy: IndexStrategy,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return index.query(query_embedding.reshape(1, -1), k)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "contiguous 2d array" not in message and "ef or m is too small" not in message:
                raise
            logger.warning("HNSW query failed with graph breadth error; rebuilding active index once: %s", exc)
            await StorageService().build_hnsw_index_async(force_rebuild=True)
            rebuilt_index = await self._load_index(embedding_dim, strategy)
            return rebuilt_index.query(query_embedding.reshape(1, -1), k)
    
    async def _lexical_retrieval(
        self,
        query: str,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[CandidatePassage]:
        """Perform BM25 lexical retrieval."""
        try:
            # Get or build BM25 index for org
            bm25_bundle = await self._get_bm25_index(org_id, source_type_filter, patient_id, constraints)
            if not bm25_bundle:
                return []
            bm25 = bm25_bundle["bm25"]
            chunk_ids = bm25_bundle["chunk_ids"]
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)
            
            # Get top K
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Fetch chunks
            db = get_database()
            filter_dict = self._retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            chunks_cursor = db.chunks.find(filter_dict)
            chunks = await chunks_cursor.to_list(length=None)
            chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks if chunk.get("chunk_id")}
            
            # Create candidate passages
            candidates = []
            for idx in top_indices:
                if idx < len(chunk_ids) and scores[idx] > 0:
                    chunk_id = chunk_ids[idx]
                    chunk = chunk_map.get(chunk_id)
                    if not chunk:
                        continue
                    candidates.append(CandidatePassage(
                        chunk_id=chunk_id,
                        doc_id=chunk.get("doc_id", ""),
                        source_type=SourceType(chunk.get("source_type", "CPG")),
                        text=chunk.get("text", chunk.get("content", "")),
                        section=chunk.get("section"),
                        offset_start=chunk.get("offset_start"),
                        offset_end=chunk.get("offset_end"),
                        metadata=chunk.get("metadata", {}),
                        token_count=chunk.get("token_count") or chunk.get("metadata", {}).get("token_count"),
                        evidence_grade_score=(chunk.get("evidence_grade") or {}).get("score") if isinstance(chunk.get("evidence_grade"), dict) else None,
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
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[BM25Okapi]:
        """Get or build BM25 index for organization."""
        cache_key = f"{org_id}_{source_type_filter}_{patient_id}_{constraints}_{settings.active_embedding_space() if settings.CLOUD_MODE else 'local'}"
        
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]
        
        try:
            # Fetch chunks
            db = get_database()
            filter_dict = self._retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            
            chunks_cursor = db.chunks.find(filter_dict, {"chunk_id": 1, "text": 1, "content": 1, "tokenized_text": 1})
            chunks = await chunks_cursor.to_list(length=None)
            
            if not chunks:
                return None
            
            # Tokenize texts
            chunk_ids: list[str] = []
            tokenized_texts = []
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                chunk_ids.append(chunk_id)
                if chunk.get("tokenized_text"):
                    tokenized_texts.append(chunk["tokenized_text"])
                else:
                    text = chunk.get("text") or chunk.get("content", "")
                    tokens = text.lower().split()
                    tokenized_texts.append(tokens)
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            self._bm25_cache[cache_key] = {"bm25": bm25, "chunk_ids": chunk_ids}

            return self._bm25_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return None

    def _retrieval_filter(
        self,
        org_id: str,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build Mongo retrieval filters from policy constraints."""
        constraints = constraints or {}
        filter_dict: Dict[str, Any] = {"org_id": org_id}

        source_policy = constraints.get("source_policy")
        if source_type_filter:
            filter_dict["source_type"] = source_type_filter.value
        elif source_policy and source_policy != "ANY":
            source_policy_map = {
                "CPG_ONLY": SourceType.CPG.value,
                "EMR_ONLY": SourceType.EMR.value,
                "LIT_ONLY": SourceType.LIT.value,
                SourceType.CPG.value: SourceType.CPG.value,
                SourceType.EMR.value: SourceType.EMR.value,
                SourceType.LIT.value: SourceType.LIT.value,
            }
            mapped_source = source_policy_map.get(str(source_policy).upper())
            if mapped_source:
                filter_dict["source_type"] = mapped_source

        # Root Cause vs Logic: patient_id was previously applied as a hard filter for
        # every retrieval request, which unintentionally excluded literature and guideline
        # evidence from patient-scoped benchmarks. The logic now only hard-filters by
        # patient_id when the request is explicitly EMR-only or the caller opts into
        # patient-scope-only retrieval; otherwise patient_id remains a ranking/context cue.
        # For mixed-source CDS runs we still keep EMR evidence patient-scoped so the
        # benchmark cannot accidentally retrieve another patient's note as if it were
        # supporting literature.
        if patient_id and (
            source_type_filter == SourceType.EMR
            or str(source_policy).upper() == "EMR_ONLY"
            or constraints.get("patient_scope_only") is True
        ):
            filter_dict["patient_id"] = patient_id
        elif patient_id:
            filter_dict["$and"] = filter_dict.get("$and", []) + [
                {"$or": [
                    {"source_type": {"$ne": SourceType.EMR.value}},
                    {"patient_id": patient_id},
                ]}
            ]

        if settings.CLOUD_MODE:
            filter_dict["embedding_space"] = settings.active_embedding_space()
            filter_dict["embedding_model"] = settings.CLOUD_EMBEDDING
            filter_dict["embedding_dim"] = settings.active_embedding_dimension()

        min_grade = constraints.get("min_evidence_grade")
        if min_grade is not None:
            try:
                min_grade_value = float(min_grade)
                filter_dict["$or"] = [
                    {"evidence_grade.score": {"$gte": min_grade_value}},
                    {"metadata.evidence_grade.score": {"$gte": min_grade_value}},
                    {"source_reliability": {"$gte": min_grade_value}},
                ]
            except (TypeError, ValueError):
                pass

        timeframe = constraints.get("timeframe")
        if isinstance(timeframe, dict):
            date_filter = {}
            if timeframe.get("start"):
                date_filter["$gte"] = timeframe["start"]
            if timeframe.get("end"):
                date_filter["$lte"] = timeframe["end"]
            if date_filter:
                filter_dict["$and"] = filter_dict.get("$and", []) + [
                    {"$or": [{"timestamp": date_filter}, {"metadata.effective_date": date_filter}]}
                ]
        elif isinstance(timeframe, str) and len(timeframe) == 4 and timeframe.isdigit():
            filter_dict["$and"] = filter_dict.get("$and", []) + [
                {"$or": [
                    {"timestamp": {"$gte": f"{timeframe}-01-01", "$lte": f"{timeframe}-12-31"}},
                    {"metadata.effective_date": {"$gte": f"{timeframe}-01-01", "$lte": f"{timeframe}-12-31"}},
                ]}
            ]

        specialties = constraints.get("specialties") or []
        if specialties:
            filter_dict["$and"] = filter_dict.get("$and", []) + [
                {"$or": [
                    {"tags": {"$in": specialties}},
                    {"metadata.specialty": {"$in": specialties}},
                    {"metadata.specialties": {"$in": specialties}},
                ]}
            ]

        return filter_dict
    
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
                    p_hat = clamp(result.get("p_hat", result.get("score", 0.0)))
                    candidates[idx].rerank_score = p_hat
                    candidates[idx].calibrated_score = p_hat
                    candidates[idx].metadata["rerank_logit"] = result.get("logit")
                    candidates[idx].metadata["calibration_version"] = result.get("calibration_version") or "identity:raw-rerank"
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return candidates
    
    def compute_fusion_scores(self, candidates: List[CandidatePassage]) -> List[CandidatePassage]:
        """Compute policy-aware fusion scores for candidates.

        Root Cause vs Logic: The previous implementation added heterogeneous raw scores
        linearly, which made cosine, BM25, and reranker probabilities look equally
        calibrated. The replacement maps every signal into a bounded feature and puts
        reranker probability on the log-odds scale before adding clinical priors.
        """
        for candidate in candidates:
            p_hat = clamp(candidate.calibrated_score or candidate.rerank_score or 0.50, 1e-6, 1.0 - 1e-6)
            rerank_log_odds = self._clip_logit(math.log(p_hat / (1.0 - p_hat)))
            dense = clamp(candidate.dense_score or 0.0)
            lexical = clamp(candidate.lexical_score or 0.0)
            recency_score = self._compute_recency_score(candidate)
            section_score = self._compute_section_score(candidate)
            source_score = self._compute_source_score(candidate)
            evidence_grade = evidence_grade_from_metadata(candidate)
            ebm_score = clamp(evidence_grade.score)
            safety_score = self._compute_safety_score(candidate)
            noise_score = self._compute_noise_score(candidate)

            raw = (
                settings.W_RERANK * rerank_log_odds
                + settings.W_DENSE * dense
                + settings.W_LEX * lexical
                + settings.W_RECENCY * recency_score
                + settings.W_SECTION * section_score
                + settings.W_SOURCE * source_score
                + settings.W_EBM * ebm_score
                + settings.SAFETY_REWARD_WEIGHT * safety_score
                - settings.W_NOISE * noise_score
            )

            candidate.recency_score = recency_score
            candidate.section_score = section_score
            candidate.source_score = source_score
            candidate.evidence_grade_score = ebm_score
            candidate.safety_score = safety_score
            candidate.noise_score = noise_score
            candidate.contradiction_score = self._compute_contradiction_score(candidate)
            candidate.fusion_score = clamp(1.0 / (1.0 + math.exp(-self._clip_logit(raw))))
        
        # Sort by fusion score
        candidates.sort(key=lambda x: x.fusion_score or 0.0, reverse=True)
        
        return candidates
    
    def _compute_recency_score(self, candidate: CandidatePassage) -> float:
        """Compute recency score (0-1)."""
        timestamp = candidate.metadata.get("timestamp") or candidate.metadata.get("effective_date")
        if not timestamp:
            return 0.5
        try:
            if isinstance(timestamp, str):
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, datetime):
                parsed = timestamp
            else:
                return 0.5
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max((datetime.now(timezone.utc) - parsed).days, 0)
            return clamp(math.exp(-age_days / max(settings.RECENCY_DECAY_DAYS, 1.0)))
        except Exception:
            return 0.5
    
    def _compute_section_score(self, candidate: CandidatePassage) -> float:
        """Compute section score (recommendations > background)."""
        section = candidate.section or ""
        section_lower = section.lower()
        
        if "recommendation" in section_lower or "guideline" in section_lower or "contraindication" in section_lower:
            return settings.SECTION_RECOMMENDATION_SCORE
        elif "background" in section_lower or "introduction" in section_lower:
            return settings.SECTION_BACKGROUND_SCORE
        else:
            return settings.SECTION_DEFAULT_SCORE
    
    def _compute_source_score(self, candidate: CandidatePassage) -> float:
        """Compute source score (CPG vs EMR weighting)."""
        if candidate.source_type == SourceType.CPG:
            return settings.SOURCE_CPG_SCORE
        elif candidate.source_type == SourceType.EMR:
            return settings.SOURCE_EMR_SCORE
        else:
            return settings.SOURCE_LIT_SCORE

    def _compute_safety_score(self, candidate: CandidatePassage) -> float:
        text = candidate.text.lower()
        terms = ["contraindicat", "allergy", "adverse", "interaction", "avoid", "renal", "pregnan", "dose"]
        return clamp(sum(1 for term in terms if term in text) * 0.16)

    def _compute_noise_score(self, candidate: CandidatePassage) -> float:
        metadata = candidate.metadata or {}
        score = 0.0
        if metadata.get("obsolete") or metadata.get("superseded"):
            score += 0.45
        if metadata.get("population_mismatch"):
            score += 0.35
        if metadata.get("weak_provenance"):
            score += 0.25
        if not candidate.doc_id or not candidate.chunk_id:
            score += 0.15
        return clamp(score)

    def _compute_contradiction_score(self, candidate: CandidatePassage) -> float:
        text = candidate.text.lower()
        terms = ["conflict", "not recommended", "avoid", "insufficient", "uncertain", "contraindicat"]
        return clamp(sum(1 for term in terms if term in text) * 0.16)

    def _clip_logit(self, value: float) -> float:
        return max(-settings.FUSION_LOGIT_CLIP, min(settings.FUSION_LOGIT_CLIP, value))
    
    def select_with_mmr(
        self,
        candidates: List[CandidatePassage],
        query_embedding: np.ndarray,
        max_chunks: Optional[int] = None,
        token_budget: Optional[int] = None,
        facets: Optional[List[ClinicalFacet]] = None
    ) -> List[CandidatePassage]:
        """Select an evidence bundle using clinical utility under token budget.

        Root Cause vs Logic: The old MMR branch used score distance as a diversity
        proxy, so near-duplicate text could be selected while clinically important
        safety passages were dropped. This greedy solver uses marginal utility,
        redundancy, safety protection, and contradiction preservation instead.
        """
        if not candidates:
            return []
        
        max_chunks = max_chunks or settings.MMR_MAX_EVIDENCE_CHUNKS
        token_budget = token_budget or settings.TOKEN_BUDGET_B

        selected: List[CandidatePassage] = []
        remaining = sorted(candidates.copy(), key=lambda item: item.fusion_score or 0.0, reverse=True)
        used_tokens = 0

        while remaining and len(selected) < max_chunks:
            best_score = -float('inf')
            best_idx = -1

            for idx, candidate in enumerate(remaining):
                tokens = self._token_count(candidate)
                if used_tokens + tokens > token_budget:
                    continue
                marginal = self._candidate_marginal_utility(candidate, selected, facets)
                utility_per_token = marginal / max(tokens, 1)
                if utility_per_token > best_score:
                    best_score = utility_per_token
                    best_idx = idx

            if best_idx >= 0:
                selected_candidate = remaining.pop(best_idx)
                selected_candidate.selected_reason = f"marginal_utility_per_token={best_score:.6f}"
                used_tokens += self._token_count(selected_candidate)
                selected.append(selected_candidate)
            else:
                break

        self._preserve_critical_safety_evidence(selected, remaining, token_budget)
        self._preserve_required_source_evidence(selected, remaining, token_budget, facets)
        return selected

    def _preserve_required_source_evidence(
        self,
        selected: List[CandidatePassage],
        remaining: List[CandidatePassage],
        token_budget: int,
        facets: Optional[List[ClinicalFacet]],
    ) -> None:
        """Protect evidence for required source-specific facets.

        Root Cause vs Logic: the benchmark showed selected bundles collapsing to
        EMR-only even when LIT evidence existed. The selector now preserves the
        strongest candidate for each required source policy represented by the
        facet contract, while still obeying the token budget.
        """
        if not facets or not remaining:
            return
        required_sources = {
            str(facet.source_policy).upper()
            for facet in facets
            if facet.required and facet.source_policy and str(facet.source_policy).upper() in {"LIT", "EMR", "CPG"}
        }
        if not required_sources:
            return
        selected_sources = {item.source_type.value for item in selected}
        current_tokens = sum(self._token_count(item) for item in selected)
        for source in sorted(required_sources - selected_sources):
            candidates = [
                item for item in remaining
                if item.source_type.value == source and item.chunk_id not in {selected_item.chunk_id for selected_item in selected}
            ]
            if not candidates:
                continue
            item = max(candidates, key=lambda c: (max((c.facet_scores or {}).values(), default=0.0), c.fusion_score or 0.0))
            tokens = self._token_count(item)
            if current_tokens + tokens <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
                item.selected_reason = f"protected_required_source={source}"
                selected.append(item)
                current_tokens += tokens
                continue
            source_counts: Dict[str, int] = {}
            for candidate in selected:
                source_counts[candidate.source_type.value] = source_counts.get(candidate.source_type.value, 0) + 1
            replaceable = [
                candidate for candidate in selected
                if candidate.source_type.value != source
                and (
                    candidate.source_type.value not in required_sources
                    or source_counts.get(candidate.source_type.value, 0) > 1
                )
            ]
            if replaceable:
                victim = min(replaceable, key=lambda candidate: candidate.fusion_score or 0.0)
                projected = current_tokens - self._token_count(victim) + tokens
                if projected <= token_budget:
                    selected.remove(victim)
                    item.selected_reason = f"protected_required_source={source}"
                    selected.append(item)
                    current_tokens = projected

    def _preserve_critical_safety_evidence(
        self,
        selected: List[CandidatePassage],
        remaining: List[CandidatePassage],
        token_budget: int,
    ) -> None:
        """Keep high-severity safety/conflict evidence from being pruned away."""
        if not remaining:
            return
        selected_ids = {item.chunk_id for item in selected}
        critical = [
            item for item in remaining
            if item.chunk_id not in selected_ids
            and ((item.safety_score or self._compute_safety_score(item)) >= 0.48
                 or (item.contradiction_score or self._compute_contradiction_score(item)) >= 0.48)
        ]
        if not critical:
            return
        current_tokens = sum(self._token_count(item) for item in selected)
        for item in sorted(critical, key=lambda c: ((c.safety_score or 0.0) + (c.contradiction_score or 0.0), c.fusion_score or 0.0), reverse=True):
            tokens = self._token_count(item)
            if current_tokens + tokens <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
                item.selected_reason = "protected_critical_safety_or_contradiction"
                selected.append(item)
                current_tokens += tokens
                return
            replaceable = [
                candidate for candidate in selected
                if (candidate.safety_score or 0.0) < 0.48 and (candidate.contradiction_score or 0.0) < 0.48
            ]
            if replaceable:
                victim = min(replaceable, key=lambda candidate: candidate.fusion_score or 0.0)
                projected = current_tokens - self._token_count(victim) + tokens
                if projected <= token_budget:
                    selected.remove(victim)
                    item.selected_reason = "protected_critical_safety_or_contradiction"
                    selected.append(item)
                    return

    def _candidate_marginal_utility(
        self,
        candidate: CandidatePassage,
        selected: List[CandidatePassage],
        facets: Optional[List[ClinicalFacet]],
    ) -> float:
        relevance = candidate.fusion_score or 0.0
        safety = candidate.safety_score or self._compute_safety_score(candidate)
        contradiction = candidate.contradiction_score or self._compute_contradiction_score(candidate)
        ebm = candidate.evidence_grade_score or evidence_grade_from_metadata(candidate).score
        redundancy = self._redundancy_penalty(candidate, selected)
        facet_gain = self._facet_gain(candidate, selected, facets)
        return (
            relevance
            + settings.SAFETY_REWARD_WEIGHT * safety
            + 0.25 * ebm
            + facet_gain
            + 0.12 * contradiction
            - settings.REDUNDANCY_PENALTY_WEIGHT * redundancy
            - settings.NOISE_PENALTY_WEIGHT * (candidate.noise_score or 0.0)
        )

    def _facet_gain(
        self,
        candidate: CandidatePassage,
        selected: List[CandidatePassage],
        facets: Optional[List[ClinicalFacet]],
    ) -> float:
        if not facets:
            return 0.0
        covered = set()
        for passage in selected:
            covered.update(name for name, score in passage.facet_scores.items() if score > 0.25)
        gain = 0.0
        for facet in facets:
            score = candidate.facet_scores.get(facet.name, 0.0)
            if score <= 0.0:
                continue
            novelty = 1.0 if facet.name not in covered else 0.35
            gain += facet.weight * score * novelty
        return gain

    def _redundancy_penalty(self, candidate: CandidatePassage, selected: List[CandidatePassage]) -> float:
        if not selected:
            return 0.0
        candidate_terms = set(candidate.text.lower().split())
        if not candidate_terms:
            return 0.0
        max_overlap = 0.0
        for passage in selected:
            selected_terms = set(passage.text.lower().split())
            if not selected_terms:
                continue
            overlap = len(candidate_terms & selected_terms) / len(candidate_terms | selected_terms)
            same_doc = 0.15 if passage.doc_id == candidate.doc_id else 0.0
            max_overlap = max(max_overlap, overlap + same_doc)
        return clamp(max_overlap)

    def _token_count(self, candidate: CandidatePassage) -> int:
        if candidate.token_count:
            return candidate.token_count
        candidate.token_count = len(self.tokenizer.encode(candidate.text))
        return candidate.token_count
    
    def build_evidence_bundle(
        self,
        passages: List[CandidatePassage],
        facet_coverage: Optional[List[FacetCoverage]] = None,
        evidence_ledger: Optional[List[EvidenceLedgerEntry]] = None,
        contradictions: Optional[List[ContradictionPair]] = None,
        policy_decision: Optional[PolicyDecision] = None
    ) -> EvidenceBundle:
        """Build evidence bundle from selected passages."""
        total_tokens = sum(self._token_count(p) for p in passages)
        
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
            coverage_ratios=coverage_ratios,
            facet_coverage=facet_coverage or [],
            evidence_ledger=evidence_ledger or [],
            contradictions=contradictions or [],
            policy_decision=policy_decision
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
