"""Lexical retrieval with a disk-backed full-corpus BM25 path.

Small development corpora keep the legacy in-memory rank_bm25 behavior. Full
TREC-CDS evaluations use an SQLite FTS5 index created by the publication corpus
preparer. SQLite FTS5's bm25() ranking keeps the lexical stage disk-backed and
avoids loading millions of PMC chunks into Python memory on every query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
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
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/-]*")


class LexicalRetriever:
    def __init__(self):
        self._memory: Dict[str, Dict[str, Any]] = {}

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

    @staticmethod
    def _fts_path(org_id: str) -> Path:
        explicit = (os.getenv("LEXICAL_FTS_PATH") or "").strip()
        if explicit:
            return Path(explicit.format(org_id=org_id))
        return CACHE_DIR / f"{org_id}_fts.sqlite"

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = []
        seen = set()
        for match in _FTS_TOKEN_RE.finditer(query.lower()):
            term = match.group(0).replace('"', '""')
            if term and term not in seen:
                seen.add(term)
                terms.append(f'"{term}"')
        # Legacy BM25 is an any-term scorer, so use OR rather than requiring all
        # clinical terms to occur in one passage.
        return " OR ".join(terms[:64])

    async def _fts_candidates(
        self,
        query: str,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
    ) -> Optional[List[CandidatePassage]]:
        path = self._fts_path(org_id)
        if not path.exists():
            return None
        match_query = self._fts_query(query)
        if not match_query:
            return []

        where = ["chunks_fts MATCH ?", "org_id = ?"]
        params: list[Any] = [match_query, org_id]
        if source_type_filter is not None:
            where.append("source_type = ?")
            params.append(source_type_filter.value)
        if patient_id:
            # Patient-scoped retrieval still allows global literature; only EMR
            # rows are narrowed to the active patient by the Mongo filter below.
            pass
        source_policy = str((constraints or {}).get("source_policy") or "ANY").upper()
        if source_type_filter is None and source_policy.endswith("_ONLY"):
            source = source_policy.removesuffix("_ONLY")
            if source in {"CPG", "EMR", "LIT", "SAFETY"}:
                where.append("source_type = ?")
                params.append(source)

        oversample = max(k, min(max(k * 20, 100), 2000))
        params.append(oversample)
        sql = (
            "SELECT chunk_id, -bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE " + " AND ".join(where) + " "
            "ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        )
        try:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("FTS5 BM25 query failed for %s: %s", path, exc)
            raise
        if not rows:
            return []

        score_map = {str(chunk_id): float(score or 0.0) for chunk_id, score in rows}
        ordered_ids = [str(chunk_id) for chunk_id, _score in rows]
        db = get_database()
        filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
        filter_dict["chunk_id"] = {"$in": ordered_ids}
        chunks = await db.chunks.find(filter_dict).to_list(length=oversample)
        chunk_map = {str(chunk.get("chunk_id")): chunk for chunk in chunks if chunk.get("chunk_id")}

        candidates: List[CandidatePassage] = []
        for chunk_id in ordered_ids:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            try:
                source_type = SourceType(chunk.get("source_type", "LIT"))
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
                    lexical_score=score_map.get(chunk_id, 0.0),
                    retrieved_by=["lexical_bm25_fts5"],
                )
            )
            if len(candidates) >= k:
                break
        return candidates

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
        # Fail closed on publication-sized corpora if the disk BM25 artifact was
        # not prepared; silently materializing millions of tokens is not valid.
        in_memory_cap = int(os.getenv("LEXICAL_IN_MEMORY_MAX_CHUNKS", "100000"))
        count = await db.chunks.count_documents(filter_dict, limit=in_memory_cap + 1)
        if count > in_memory_cap:
            raise RuntimeError(
                f"Disk BM25 index {self._fts_path(org_id)} is required for {count}+ chunks; "
                "run eval/scripts/prepare_full_trec_runtime.py"
            )
        chunks = await db.chunks.find(
            filter_dict,
            {"chunk_id": 1, "text": 1, "content": 1, "tokenized_text": 1},
        ).to_list(length=in_memory_cap)
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
            full_corpus = await self._fts_candidates(
                query, org_id, k, source_type_filter, patient_id, constraints
            )
            if full_corpus is not None:
                return full_corpus

            bundle = await self._get_bundle(org_id, source_type_filter, patient_id, constraints)
            if not bundle:
                return []
            bm25 = bundle["bm25"]
            chunk_ids = bundle["chunk_ids"]
            scores = bm25.get_scores(query.lower().split())
            top_indices = np.argsort(scores)[::-1][:k]
            selected_ids = [chunk_ids[int(idx)] for idx in top_indices if idx < len(chunk_ids) and scores[idx] > 0]
            if not selected_ids:
                return []
            db = get_database()
            filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
            filter_dict["chunk_id"] = {"$in": selected_ids}
            chunks = await db.chunks.find(filter_dict).to_list(length=len(selected_ids))
            chunk_map = {c["chunk_id"]: c for c in chunks if c.get("chunk_id")}
            candidates = []
            for idx in top_indices:
                if idx >= len(chunk_ids) or scores[idx] <= 0:
                    continue
                chunk = chunk_map.get(chunk_ids[int(idx)])
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
                        lexical_score=float(scores[int(idx)]),
                        retrieved_by=["lexical"],
                    )
                )
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.error("Lexical retrieval failed: %s", exc)
            return []
