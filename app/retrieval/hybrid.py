"""Stage-1 hybrid retrieval + Stage-2 rerank orchestration helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.hints import apply_hints
from app.retrieval.lexical import LexicalRetriever
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage, EvidenceBundle
from app.schemas.facets import ClinicalFacet, FacetCoverage
from app.schemas.evidence import ContradictionPair, EvidenceLedgerEntry, PolicyDecision
from app.schemas.traces import RerankTrace, RetrievalTrace
from app.scoring.calibrate import apply_calibration, clamp, get_calibration_store
from app.scoring.coverage import score_passage_facets
from app.scoring.fusion import compute_fusion_scores
from app.scoring.utility import select_bundle, token_count
from app.services.adapters.reranker import RerankerClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Two-stage hybrid retriever with calibrated reranking."""

    def __init__(self, reranker_client: Optional[RerankerClient] = None):
        self.dense = DenseRetriever()
        self.lexical = LexicalRetriever()
        self.reranker_client = reranker_client
        self.calibration = get_calibration_store()

    def _union(
        self,
        dense_candidates: List[CandidatePassage],
        lexical_candidates: List[CandidatePassage],
    ) -> List[CandidatePassage]:
        merged: Dict[str, CandidatePassage] = {}
        for candidate in dense_candidates:
            merged[candidate.chunk_id] = candidate
        for candidate in lexical_candidates:
            if candidate.chunk_id in merged:
                merged[candidate.chunk_id].lexical_score = candidate.lexical_score
                tags = set(merged[candidate.chunk_id].retrieved_by) | set(candidate.retrieved_by)
                merged[candidate.chunk_id].retrieved_by = sorted(tags)
            else:
                merged[candidate.chunk_id] = candidate
        return list(merged.values())

    def _normalize(self, candidates: List[CandidatePassage]) -> None:
        dense_scores = [c.dense_score for c in candidates if c.dense_score is not None]
        lexical_scores = [c.lexical_score for c in candidates if c.lexical_score is not None]
        if dense_scores and max(dense_scores) > min(dense_scores):
            lo, hi = min(dense_scores), max(dense_scores)
            for c in candidates:
                if c.dense_score is not None:
                    c.dense_score = (c.dense_score - lo) / (hi - lo)
        if lexical_scores and max(lexical_scores) > min(lexical_scores):
            lo, hi = min(lexical_scores), max(lexical_scores)
            for c in candidates:
                if c.lexical_score is not None:
                    c.lexical_score = (c.lexical_score - lo) / (hi - lo)

    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        org_id: str,
        top_k: Optional[int] = None,
        source_type_filter: Optional[SourceType] = None,
        patient_id: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[CandidatePassage], RetrievalTrace]:
        expanded, constraints, k, hint_source = apply_hints(query, constraints, hints, top_k)
        effective_source = source_type_filter or hint_source

        dense = await self.dense.retrieve(
            query_embedding, org_id, k, effective_source, patient_id, constraints
        )
        lexical = []
        if settings.ENABLE_BM25:
            lexical = await self.lexical.retrieve(
                expanded, org_id, k, effective_source, patient_id, constraints
            )
        all_candidates = self._union(dense, lexical)

        # Source-balanced probes for mixed-source CDS
        if effective_source is None and str((constraints or {}).get("source_policy") or "ANY").upper() == "ANY":
            for source_type in (SourceType.LIT, SourceType.EMR, SourceType.SAFETY):
                source_constraints = dict(constraints or {})
                source_constraints["source_policy"] = f"{source_type.value}_ONLY"
                d = await self.dense.retrieve(
                    query_embedding, org_id, max(1, k // 2), source_type, patient_id, source_constraints
                )
                l = []
                if settings.ENABLE_BM25:
                    l = await self.lexical.retrieve(
                        expanded, org_id, max(1, k // 2), source_type, patient_id, source_constraints
                    )
                all_candidates = self._union(all_candidates, self._union(d, l))

        self._normalize(all_candidates)
        all_candidates.sort(key=lambda c: (c.dense_score or 0.0, c.lexical_score or 0.0), reverse=True)
        # TopK' for reranking pool
        pool = all_candidates[: max(k, settings.CANDIDATE_K)]
        trace = RetrievalTrace(
            dense_count=len(dense),
            lexical_count=len(lexical),
            union_count=len(pool),
            hints=hints or {},
            candidates=[
                {
                    "chunk_id": c.chunk_id,
                    "source_type": c.source_type.value,
                    "dense_score": c.dense_score,
                    "lexical_score": c.lexical_score,
                    "retrieved_by": c.retrieved_by,
                }
                for c in pool[:40]
            ],
        )
        return pool, trace

    async def rerank(
        self,
        query: str,
        candidates: List[CandidatePassage],
    ) -> Tuple[List[CandidatePassage], RerankTrace]:
        if not self.reranker_client or not candidates:
            return candidates, RerankTrace(calibration_version=self.calibration.version)
        try:
            passages = [c.text for c in candidates]
            raw = await self.reranker_client.rerank(query, passages, return_logits=True)
            calibrated = apply_calibration(raw, self.calibration)
            for result in calibrated:
                idx = result["index"]
                if idx < len(candidates):
                    p_hat = clamp(result.get("p_hat", 0.0))
                    candidates[idx].rerank_score = p_hat
                    candidates[idx].calibrated_score = p_hat
                    candidates[idx].metadata["rerank_logit"] = result.get("logit")
                    candidates[idx].metadata["calibration_version"] = result.get("calibration_version")
            candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
            version = calibrated[0].get("calibration_version") if calibrated else self.calibration.version
            trace = RerankTrace(
                calibration_version=version,
                scores=[
                    {
                        "chunk_id": c.chunk_id,
                        "p_hat": c.calibrated_score,
                        "logit": c.metadata.get("rerank_logit"),
                    }
                    for c in candidates[:40]
                ],
            )
            return candidates, trace
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed: %s", exc)
            return candidates, RerankTrace(calibration_version="identity:rerank-error")

    def compute_fusion_scores(self, candidates: List[CandidatePassage]) -> List[CandidatePassage]:
        """Legacy-compatible fusion entrypoint."""
        return compute_fusion_scores(candidates)

    def select_with_mmr(
        self,
        candidates: List[CandidatePassage],
        query_embedding=None,
        max_chunks: Optional[int] = None,
        token_budget: Optional[int] = None,
        facets: Optional[List[ClinicalFacet]] = None,
    ) -> List[CandidatePassage]:
        """Legacy-compatible budgeted selection entrypoint."""
        return select_bundle(
            candidates,
            facets=facets,
            max_chunks=max_chunks,
            token_budget=token_budget,
        )

    def fuse_and_select(
        self,
        candidates: List[CandidatePassage],
        facets: Optional[List[ClinicalFacet]] = None,
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> List[CandidatePassage]:
        for candidate in candidates:
            if facets:
                score_passage_facets(candidate, facets)
        fused = compute_fusion_scores(candidates)
        return select_bundle(fused, facets=facets, agent_weights=agent_weights)

    def build_bundle(
        self,
        passages: List[CandidatePassage],
        facet_coverage: Optional[List[FacetCoverage]] = None,
        evidence_ledger: Optional[List[EvidenceLedgerEntry]] = None,
        contradictions: Optional[List[ContradictionPair]] = None,
        policy_decision: Optional[PolicyDecision] = None,
    ) -> EvidenceBundle:
        total_tokens = sum(token_count(p) for p in passages)
        cpg_count = sum(1 for p in passages if p.source_type == SourceType.CPG)
        emr_count = sum(1 for p in passages if p.source_type == SourceType.EMR)
        lit_count = sum(1 for p in passages if p.source_type == SourceType.LIT)
        safety_count = sum(1 for p in passages if p.source_type == SourceType.SAFETY)
        n = len(passages) or 1
        return EvidenceBundle(
            passages=passages,
            total_tokens=total_tokens,
            cpg_count=cpg_count,
            emr_count=emr_count,
            lit_count=lit_count,
            safety_count=safety_count,
            coverage_ratios={
                "cpg_ratio": cpg_count / n if passages else 0.0,
                "emr_ratio": emr_count / n if passages else 0.0,
                "lit_ratio": lit_count / n if passages else 0.0,
                "safety_ratio": safety_count / n if passages else 0.0,
            },
            facet_coverage=facet_coverage or [],
            evidence_ledger=evidence_ledger or [],
            contradictions=contradictions or [],
            policy_decision=policy_decision,
        )


# Backward-compatible alias used by older imports
RetrievalPipeline = HybridRetriever
