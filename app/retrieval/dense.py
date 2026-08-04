"""Dense ANN retrieval via hybrid HNSW ∪ IVF."""

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


class DenseRetriever:
    def __init__(self):
        self.index = HybridIndex()

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
            chunk_ids, score_map = self.index.query(query_embedding, k)
            if not chunk_ids:
                return []
            filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            filter_dict["chunk_id"] = {"$in": chunk_ids}
            db = get_database()
            chunks = await db.chunks.find(filter_dict).to_list(length=None)
            candidates = []
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                try:
                    source_type = SourceType(chunk.get("source_type", "CPG"))
                except ValueError:
                    source_type = SourceType.LIT
                candidates.append(
                    CandidatePassage(
                        chunk_id=chunk_id,
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
                        dense_score=score_map.get(chunk_id, 0.0),
                        retrieved_by=["dense"],
                    )
                )
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense retrieval failed: %s", exc)
            return []
