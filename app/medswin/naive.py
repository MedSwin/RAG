"""Naive-RAG baseline: embed query → dense top-K → generate.

This is the control pipeline for MedSwin evaluation. It reuses the same
embedding client, ANN index, Mongo chunk store, org/patient filters, and
LLM backend as the full system. It deliberately omits BM25, reranking,
calibration, fusion, utility selection, MAC agents, the sufficiency gate,
retrieve-more, and hint expansion.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import settings
from app.core.database import get_database
from app.medswin.ledger import build_retrieval_ledger
from app.repositories.sessions import SessionRepository
from app.repositories.traces import TraceRepository
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import retrieval_filter
from app.retrieval.hybrid import HybridRetriever
from app.schemas.enums import PolicyAction, SourceType
from app.schemas.evidence import (
    AnswerProvenance,
    CandidatePassage,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceLedgerEntry,
    PolicyDecision,
)
from app.schemas.sessions import Session
from app.schemas.traces import AgentMessage, AuditTrace, ChatResponse, RetrievalTrace, ToolCall
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient
from app.services.medswin.governance import build_citation

logger = logging.getLogger(__name__)

PIPELINE_ID = "naive_rag"
NAIVE_POLICY_REASON = (
    "Naive RAG does not apply an evidence-sufficiency gate. "
    "Generation always proceeds from dense top-K (or from an empty context)."
)


def _cosine(query: np.ndarray, vector: Sequence[float]) -> float:
    left = np.asarray(query, dtype=np.float32).reshape(-1)
    right = np.asarray(vector, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _chunk_to_passage(chunk: Dict[str, Any], dense_score: float, retrieved_by: str) -> CandidatePassage:
    try:
        source_type = SourceType(chunk.get("source_type", "LIT"))
    except ValueError:
        source_type = SourceType.LIT
    grade = chunk.get("evidence_grade") if isinstance(chunk.get("evidence_grade"), dict) else {}
    return CandidatePassage(
        chunk_id=chunk.get("chunk_id", ""),
        doc_id=chunk.get("doc_id", ""),
        source_type=source_type,
        text=chunk.get("text") or chunk.get("content") or "",
        section=chunk.get("section"),
        offset_start=chunk.get("offset_start"),
        offset_end=chunk.get("offset_end"),
        metadata=chunk.get("metadata") or {},
        token_count=chunk.get("token_count") or (chunk.get("metadata") or {}).get("token_count"),
        evidence_grade_score=grade.get("score"),
        dense_score=dense_score,
        selected_reason="naive_dense_topk",
        retrieved_by=[retrieved_by],
    )


def _truncate_context(passages: List[CandidatePassage], max_chars: int) -> List[CandidatePassage]:
    kept: List[CandidatePassage] = []
    used = 0
    for passage in passages:
        text = passage.text or ""
        if used and used + len(text) > max_chars:
            break
        kept.append(passage)
        used += len(text)
    return kept or passages[:1]


def compare_responses(
    naive: ChatResponse,
    medswin: ChatResponse,
    *,
    naive_ms: Optional[float] = None,
    medswin_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Side-by-side retrieval/generation diff for one query."""
    naive_ids = [p.chunk_id for p in naive.evidence_bundle.passages]
    medswin_ids = [p.chunk_id for p in medswin.evidence_bundle.passages]
    naive_set, medswin_set = set(naive_ids), set(medswin_ids)
    overlap = sorted(naive_set & medswin_set)
    naive_decision = naive.policy_decision
    medswin_decision = medswin.policy_decision
    return {
        "naive_chunk_ids": naive_ids,
        "medswin_chunk_ids": medswin_ids,
        "overlap_chunk_ids": overlap,
        "naive_only_chunk_ids": sorted(naive_set - medswin_set),
        "medswin_only_chunk_ids": sorted(medswin_set - naive_set),
        "overlap_count": len(overlap),
        "jaccard": (len(overlap) / len(naive_set | medswin_set)) if (naive_set or medswin_set) else 0.0,
        "naive_passage_count": len(naive_ids),
        "medswin_passage_count": len(medswin_ids),
        "naive_always_generated": True,
        "medswin_abstained": bool(medswin_decision and not medswin_decision.passed),
        "medswin_policy_action": medswin_decision.action.value if medswin_decision else None,
        "naive_policy_action": naive_decision.action.value if naive_decision else None,
        "naive_backend": naive.retrieval_backend,
        "medswin_backend": medswin.retrieval_backend,
        "timing_ms": {
            "naive": naive_ms if naive_ms is not None else naive.timing_ms.get("total"),
            "medswin": medswin_ms if medswin_ms is not None else medswin.timing_ms.get("total"),
        },
    }


class NaiveRAGOrchestrator:
    """Control RAG used to isolate MedSwin design effects."""

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        llm_client: Optional[LLMClient] = None,
        dense_retriever: Optional[DenseRetriever] = None,
    ):
        cloud_model = settings.CLOUD_MODEL if settings.CLOUD_MODE else None
        self.embedding_client = embedding_client or EmbeddingClient(settings.active_embedding_url())
        self.llm = llm_client or LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
        self.dense = dense_retriever or DenseRetriever()
        self.bundle_builder = HybridRetriever(reranker_client=None)
        self.session_repo = SessionRepository()
        self.trace_repo = TraceRepository()

    async def chat(
        self,
        query: str,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        constraints = constraints or {}
        raw_k = settings.NAIVE_TOP_K if top_k is None else top_k
        if top_k is None and constraints.get("top_k") is not None:
            raw_k = constraints.get("top_k")
        k = max(1, min(int(raw_k), settings.MAX_TOP_K))

        trace_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
            await self.session_repo.create(Session(session_id=session_id, user_id=user_id, org_id=org_id), org_id)
        else:
            await self.session_repo.update_last_active(session_id, org_id)

        trace = AuditTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            query=query,
            patient_id=patient_id,
            pipeline=PIPELINE_ID,
        )

        try:
            embed_started = time.perf_counter()
            embeddings = await self.embedding_client.embed([query])
            embed_ms = (time.perf_counter() - embed_started) * 1000.0
            query_embedding = embeddings[0] if embeddings else None
            if query_embedding is None:
                raise RuntimeError("Embedding service returned no vector for the query")

            retrieve_started = time.perf_counter()
            passages, backend, retrieval_trace = await self._retrieve(
                query_embedding, org_id, k, patient_id, constraints
            )
            retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0
            trace.retrieval_traces.append(retrieval_trace)
            trace.retrieval_backend = backend
            trace.tool_calls.append(
                ToolCall(
                    tool_name="retrieval.naive_dense",
                    parameters={"top_k": k, "backend": backend},
                    result={"count": len(passages)},
                )
            )

            diagnosis = await self._diagnose_corpus(org_id, patient_id, constraints)
            if not passages and self._is_infrastructure_gap(diagnosis, backend):
                note = self._infrastructure_answer(backend, diagnosis)
                degraded = {"no_embeddings": True} if diagnosis.get("embedded_count") == 0 else {"empty_index": True}
                return await self._respond(
                    trace,
                    org_id,
                    passages=[],
                    backend=backend,
                    answer=note,
                    notes=note,
                    started=started,
                    embed_ms=embed_ms,
                    retrieve_ms=retrieve_ms,
                    generate_ms=0.0,
                    degraded=degraded,
                )

            packed = _truncate_context(passages, settings.NAIVE_MAX_CONTEXT_CHARS)
            generate_started = time.perf_counter()
            answer = await self._generate(query, packed)
            generate_ms = (time.perf_counter() - generate_started) * 1000.0
            return await self._respond(
                trace,
                org_id,
                passages=packed,
                backend=backend,
                answer=answer,
                notes=NAIVE_POLICY_REASON,
                started=started,
                embed_ms=embed_ms,
                retrieve_ms=retrieve_ms,
                generate_ms=generate_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Naive RAG failed: %s", exc, exc_info=True)
            return await self._respond(
                trace,
                org_id,
                passages=[],
                backend="error",
                answer=f"Naive RAG failed: {exc}",
                notes=str(exc),
                started=started,
                embed_ms=0.0,
                retrieve_ms=0.0,
                generate_ms=0.0,
                degraded={"error": True},
            )

    async def _retrieve(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        patient_id: Optional[str],
        constraints: Dict[str, Any],
    ) -> Tuple[List[CandidatePassage], str, RetrievalTrace]:
        source_type_filter = None
        if constraints.get("guideline_only"):
            source_type_filter = SourceType.CPG
        candidates = await self.dense.retrieve(
            query_embedding,
            org_id,
            k,
            source_type_filter,
            patient_id,
            constraints,
        )
        backend = "ann"
        skipped_dim = 0
        if not candidates and settings.NAIVE_ENABLE_MONGO_FALLBACK:
            candidates, skipped_dim = await self._mongo_cosine(
                query_embedding, org_id, k, source_type_filter, patient_id, constraints
            )
            if candidates:
                backend = "mongo_cosine"
            elif skipped_dim:
                backend = "dim_mismatch"
            else:
                backend = "empty"
        elif not candidates:
            backend = "empty"

        candidates.sort(key=lambda item: item.dense_score or 0.0, reverse=True)
        selected = candidates[:k]
        trace = RetrievalTrace(
            dense_count=len(candidates),
            lexical_count=0,
            union_count=len(selected),
            hints={"pipeline": PIPELINE_ID, "backend": backend, "top_k": k},
            candidates=[
                {
                    "chunk_id": item.chunk_id,
                    "source_type": item.source_type.value,
                    "dense_score": item.dense_score,
                    "retrieved_by": item.retrieved_by,
                }
                for item in selected
            ],
        )
        return selected, backend, trace

    async def _mongo_cosine(
        self,
        query_embedding: np.ndarray,
        org_id: str,
        k: int,
        source_type_filter: Optional[SourceType],
        patient_id: Optional[str],
        constraints: Dict[str, Any],
    ) -> Tuple[List[CandidatePassage], int]:
        """Local-dev fallback when the ANN index is missing or empty.

        Benchmarks that already have a provenance-stamped HNSW/IVF index never
        take this path. The scan is capped so a full TREC corpus cannot be
        loaded into memory during a smoke prompt.
        """
        filter_dict = retrieval_filter(org_id, source_type_filter, patient_id, constraints)
        filter_dict["embedding"] = {"$exists": True, "$ne": None}
        try:
            db = get_database()
            chunks = await db.chunks.find(filter_dict).to_list(length=settings.NAIVE_MONGO_SCAN_LIMIT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Naive Mongo cosine fallback failed: %s", exc)
            return [], 0

        scored: List[CandidatePassage] = []
        skipped_dim = 0
        query_dim = int(np.asarray(query_embedding).reshape(-1).size)
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            if len(embedding) != query_dim:
                skipped_dim += 1
                continue
            score = _cosine(query_embedding, embedding)
            scored.append(_chunk_to_passage(chunk, score, "mongo_cosine"))
        scored.sort(key=lambda item: item.dense_score or 0.0, reverse=True)
        return scored[:k], skipped_dim

    async def _generate(self, query: str, passages: List[CandidatePassage]) -> str:
        if passages:
            context = "\n\n".join(
                f"[{idx}] chunk_id={passage.chunk_id} doc_id={passage.doc_id}\n{passage.text}"
                for idx, passage in enumerate(passages, start=1)
            )
            instruction = (
                "Answer the question using the retrieved passages. "
                "If the passages are weak or incomplete, still give your best answer."
            )
        else:
            context = "(no passages retrieved)"
            instruction = (
                "No passages were retrieved. Answer the question from your own knowledge."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a naive retrieval-augmented assistant. "
                    "Do not apply evidence-sufficiency rules, specialist critique, or abstention. "
                    f"{instruction}"
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nPassages:\n{context}",
            },
        ]
        response = await self.llm.call_llm(messages, temperature=0.2)
        return str(response.get("content") or "").strip() or "No answer was generated."

    async def _diagnose_corpus(
        self,
        org_id: str,
        patient_id: Optional[str],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        filter_dict = retrieval_filter(org_id, None, patient_id, constraints)
        try:
            db = get_database()
            chunk_count = await db.chunks.count_documents(filter_dict)
            embedded_count = await db.chunks.count_documents(
                {**filter_dict, "embedding": {"$exists": True, "$type": "array", "$ne": []}}
            )
            return {
                "chunk_count": chunk_count,
                "embedded_count": embedded_count,
                "index_exists": Path(settings.HNSW_INDEX_PATH).exists(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Naive corpus diagnosis failed: %s", exc)
            return {
                "chunk_count": None,
                "embedded_count": None,
                "index_exists": Path(settings.HNSW_INDEX_PATH).exists(),
                "error": str(exc),
            }

    @staticmethod
    def _is_infrastructure_gap(diagnosis: Dict[str, Any], backend: str) -> bool:
        if backend == "dim_mismatch":
            return True
        chunk_count = diagnosis.get("chunk_count")
        embedded_count = diagnosis.get("embedded_count")
        if isinstance(chunk_count, int) and chunk_count > 0 and embedded_count == 0:
            return True
        return False

    @staticmethod
    def _infrastructure_answer(backend: str, diagnosis: Dict[str, Any]) -> str:
        chunks = diagnosis.get("chunk_count")
        embedded = diagnosis.get("embedded_count")
        if backend == "dim_mismatch":
            return (
                "Naive RAG did not retrieve any passages: stored embeddings do not match "
                "the active query embedding dimension. Rebuild embeddings and the ANN index "
                "with POST /api/v1/storage/embeddings/refresh then /api/v1/storage/index/build."
            )
        return (
            f"Naive RAG did not retrieve any passages: org has {chunks} chunks but "
            f"{embedded} embeddings. Ingest attaches vectors when the embedding client "
            "is reachable; otherwise call POST /api/v1/storage/embeddings/refresh and "
            "POST /api/v1/storage/index/build. This is not a valid naive-RAG run."
        )

    async def _respond(
        self,
        trace: AuditTrace,
        org_id: str,
        *,
        passages: List[CandidatePassage],
        backend: str,
        answer: str,
        notes: str,
        started: float,
        embed_ms: float,
        retrieve_ms: float,
        generate_ms: float,
        degraded: Optional[Dict[str, bool]] = None,
    ) -> ChatResponse:
        if answer and not any(message.content == answer for message in trace.messages):
            trace.messages.append(AgentMessage(role="user", agent_id="naive_rag", content=trace.query))
            trace.messages.append(AgentMessage(role="assistant", agent_id="naive_rag", content=answer))
        decision = PolicyDecision(
            passed=True,
            action=PolicyAction.ACCEPT,
            reason=notes,
        )
        ledger = self._naive_ledger(passages)
        bundle = self.bundle_builder.build_bundle(
            passages,
            evidence_ledger=ledger,
            policy_decision=decision,
        )
        citations = [build_citation(passage) for passage in passages]
        provenance = AnswerProvenance(
            statements=[{"chunk_id": p.chunk_id, "use": "naive_context"} for p in passages],
            cited_chunk_ids=[p.chunk_id for p in passages],
        )
        timing = {
            "embed": round(embed_ms, 2),
            "retrieve": round(retrieve_ms, 2),
            "generate": round(generate_ms, 2),
            "total": round((time.perf_counter() - started) * 1000.0, 2),
        }
        if degraded:
            trace.degraded_mode.update(degraded)
        trace.evidence_bundle = bundle
        trace.evidence_ledger = ledger
        trace.policy_decisions.append(decision)
        trace.answer_provenance = provenance
        trace.final_answer = answer
        trace.citations = citations
        trace.timing_ms = timing
        trace.retrieval_backend = backend
        trace.completed_at = datetime.now(timezone.utc)
        try:
            await self.trace_repo.create(trace, org_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Naive trace persist failed: %s", exc)
            if degraded is None:
                degraded = {}
            degraded["trace_persist"] = True
        return ChatResponse(
            answer=answer,
            evidence_bundle=bundle,
            safety_notes=notes,
            trace_id=trace.trace_id,
            degraded_mode=degraded or {},
            uncertainty_level="high" if (degraded or not passages) else "ungated",
            citations=citations,
            policy_decision=decision,
            evidence_ledger=ledger,
            answer_provenance=provenance,
            retrieval_traces=trace.retrieval_traces,
            pipeline=PIPELINE_ID,
            retrieval_backend=backend,
            timing_ms=timing,
        )

    def _naive_ledger(self, passages: List[CandidatePassage]) -> List[EvidenceLedgerEntry]:
        if not passages:
            return []
        for passage in passages:
            if passage.calibrated_score is None:
                passage.calibrated_score = passage.dense_score
            if passage.fusion_score is None:
                passage.fusion_score = passage.dense_score
        # Reuse the retrieval ledger shape so the eval harness can score
        # evidence_doc_recall / citation metrics without MedSwin claims.
        ledger = build_retrieval_ledger(passages, facets=[])
        for entry, passage in zip(ledger, passages):
            entry.agent_id = PIPELINE_ID
            entry.calibrated_relevance = passage.dense_score or 0.0
            entry.fusion_score = passage.dense_score or 0.0
            snippet = " ".join((passage.text or "").split())[:240]
            if snippet:
                entry.claims = [
                    EvidenceClaim(
                        facet="ungated",
                        claim=snippet,
                        chunk_id=passage.chunk_id,
                        confidence=passage.dense_score or 0.0,
                        calibrated_relevance=passage.dense_score or 0.0,
                        agent_id=PIPELINE_ID,
                    )
                ]
        return ledger
