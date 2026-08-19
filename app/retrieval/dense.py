"""Dense ANN retrieval via hybrid HNSW ∪ IVF with small-scope exact recovery."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.database import get_database
from app.indexes.hybrid import HybridIndex
from app.retrieval.filters import retrieval_filter
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage

logger = logging.getLogger(__name__)

# A global ANN index cannot apply Mongo metadata filters before nearest-neighbor
# selection. For tiny scoped corpora (notably one TREC patient's EMR chunks), an
# exact scan is both cheap and necessary: otherwise a global top-40 dominated by
# 1.25M literature chunks can be filtered down to zero EMR passages.
EXACT_FILTER_SCAN_MAX = 5_000


def _cosine(query: np.ndarray, vector: List[float]) -> float:
    left = np.asarray(query, dtype=np.float32).reshape(-1)
    right = np.asarray(vector, dtype=np.float32).reshape(-1)
    if left.size == 0 or left.size != right.size:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _passage(chunk: Dict[str, Any], score: float, retrieved_by: str) -> CandidatePassage:
    try:
        source_type = SourceType(chunk.get("source_type", "CPG"))
    except ValueError:
        source_type = SourceType.LIT
    return CandidatePassage(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk.get("doc_id", ""),
        source_type=source_type,
        text=chunk.get("text", chunk.get("content", "")),
        section=chunk.get("section"),
        offset_start=chunk.get("offset_start"),
        offset_end=chunk.get("offset_end"),
        metadata=chunk.get("metadata", {}),
        token_count=chunk.get("token_count") or chunk.get("metadata", {}).get("token_count"),
        evidence_grade_score=(chunk.get("evidence_grade") or {}).get("score")
        if isinstance(chunk.get("evidence_grade"), dict)
        else None,
        dense_score=score,
        retrieved_by=[retrieved_by],
    )


class DenseRetriever:
    def __init__(self):
        self.index = HybridIndex()

    async def _small_filtered_exact(
        self,
        query_embedding: np.ndarray,
        filter_dict: Dict[str, Any],
        k: int,
    ) -> List[CandidatePassage]:
        db = get_database()
        # Use a capped count so this path can never turn a publication-size LIT
        # filter into a Mongo full-vector scan.
        count = await db.chunks.count_documents(filter_dict, limit=EXACT_FILTER_SCAN_MAX + 1)
        if count <= 0 or count > EXACT_FILTER_SCAN_MAX:
            return []
        chunks = await db.chunks.find(
            {**filter_dict, "embedding": {"$exists": True, "$type": "array", "$ne": []}},
            {
                "chunk_id": 1,
                "doc_id": 1,
                "source_type": 1,
                "text": 1,
                "content": 1,
                "section": 1,
                "offset_start": 1,
                "offset_end": 1,
                "metadata": 1,
                "token_count": 1,
                "evidence_grade": 1,
                "embedding": 1,
            },
        ).to_list(length=EXACT_FILTER_SCAN_MAX)
        scored: List[CandidatePassage] = []
        query_dim = int(np.asarray(query_embedding).reshape(-1).size)
        for chunk in chunks:
            vector = chunk.get("embedding") or []
            if len(vector) != query_dim:
                continue
            scored.append(_passage(chunk, _cosine(query_embedding, vector), "dense_exact_filtered"))
        scored.sort(key=lambda item: item.dense_score or 0.0, reverse=True)
        return scored[:k]

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[CandidatePassage]:
        try:
            # Oversample before metadata filtering. This improves filtered LIT
            # retrieval without changing the public top-k result contract.
            ann_k = max(k, min(k * 4, 500)) if source_type_filter else k
            chunk_ids, score_map = self.index.query(query_embedding, ann_k)
            filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            candidates: List[CandidatePassage] = []
            if chunk_ids:
                db = get_database()
                query_filter = dict(filter_dict)
                query_filter["chunk_id"] = {"$in": chunk_ids}
                chunks = await db.chunks.find(query_filter).to_list(length=ann_k)
                candidates.extend(
                    _passage(chunk, score_map.get(chunk["chunk_id"], 0.0), "dense")
                    for chunk in chunks
                )

            # Source-balanced MAC probes need patient EMR / small CPG / safety
            # pools even when those chunks cannot appear in the global ANN top-k.
            if source_type_filter in {SourceType.EMR, SourceType.CPG, SourceType.SAFETY} and len(candidates) < k:
                exact = await self._small_filtered_exact(query_embedding, filter_dict, k)
                by_id = {candidate.chunk_id: candidate for candidate in candidates}
                for candidate in exact:
                    existing = by_id.get(candidate.chunk_id)
                    if existing is None or (candidate.dense_score or 0.0) > (existing.dense_score or 0.0):
                        by_id[candidate.chunk_id] = candidate
                candidates = list(by_id.values())

            candidates.sort(key=lambda item: item.dense_score or 0.0, reverse=True)
            return candidates[:k]
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense retrieval failed: %s", exc)
            return []
