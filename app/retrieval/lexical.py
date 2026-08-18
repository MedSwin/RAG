"""BM25 lexical retrieval with disk-backed corpus cache."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query_tokens):
            query = set(query_tokens)
            return np.array(
                [float(len(query & set(doc))) / max(len(query), 1) for doc in self.corpus],
                dtype=np.float32,
            )

from app.core.config import settings
from app.core.database import get_database
from app.retrieval.filters import retrieval_filter
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage

logger = logging.getLogger(__name__)

CACHE_DIR = Path(settings.DATA_DIR) / "bm25"


class LexicalRetriever:
    def __init__(self):
        self._memory: Dict[str, Dict[str, Any]] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        org_id: str,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "org_id": org_id,
            "source": source_type_filter.value if source_type_filter else None,
            "patient_id": patient_id,
            "constraints": constraints or {},
            "space": settings.active_embedding_space() if settings.CLOUD_MODE else "local",
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return f"{org_id}_{digest}"

    async def _get_bundle(
        self,
        org_id: str,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        key = self._cache_key(org_id, source_type_filter, patient_id, constraints)
        if key in self._memory:
            return self._memory[key]
        disk_path = CACHE_DIR / f"{key}.pkl"
        if disk_path.exists():
            try:
                bundle = pickle.loads(disk_path.read_bytes())
                self._memory[key] = bundle
                return bundle
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 disk cache read failed: %s", exc)

        db = get_database()
        filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
        chunks = await db.chunks.find(
            filter_dict,
            {"chunk_id": 1, "text": 1, "content": 1, "tokenized_text": 1},
        ).to_list(length=None)
        if not chunks:
            return None
        chunk_ids: List[str] = []
        tokenized = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            chunk_ids.append(cid)
            if chunk.get("tokenized_text"):
                tokenized.append(chunk["tokenized_text"])
            else:
                text = chunk.get("text") or chunk.get("content", "")
                tokenized.append(text.lower().split())
        bm25 = BM25Okapi(tokenized)
        bundle = {"bm25": bm25, "chunk_ids": chunk_ids}
        self._memory[key] = bundle
        try:
            disk_path.write_bytes(pickle.dumps(bundle))
        except Exception as exc:  # noqa: BLE001
            logger.debug("BM25 disk cache write failed: %s", exc)
        return bundle

    async def retrieve(
        self,
        query: str,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[CandidatePassage]:
        try:
            bundle = await self._get_bundle(org_id, source_type_filter, patient_id, constraints)
            if not bundle:
                return []
            bm25 = bundle["bm25"]
            chunk_ids = bundle["chunk_ids"]
            scores = bm25.get_scores(query.lower().split())
            top_indices = np.argsort(scores)[::-1][:k]
            db = get_database()
            filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            chunks = await db.chunks.find(filter_dict).to_list(length=None)
            chunk_map = {c["chunk_id"]: c for c in chunks if c.get("chunk_id")}
            candidates = []
            for idx in top_indices:
                if idx >= len(chunk_ids) or scores[idx] <= 0:
                    continue
                chunk = chunk_map.get(chunk_ids[idx])
                if not chunk:
                    continue
                try:
                    source_type = SourceType(chunk.get("source_type", "CPG"))
                except ValueError:
                    source_type = SourceType.LIT
                candidates.append(
                    CandidatePassage(
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
                        lexical_score=float(scores[idx]),
                        retrieved_by=["lexical"],
                    )
                )
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.error("Lexical retrieval failed: %s", exc)
            return []
