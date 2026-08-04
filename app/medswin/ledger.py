"""Claim-level evidence ledger merge."""

from __future__ import annotations

from typing import Dict, List, Optional

from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity
from app.schemas.evidence import (
    CandidatePassage,
    ContradictionPair,
    EvidenceClaim,
    EvidenceLedgerEntry,
)
from app.schemas.facets import ClinicalFacet
from app.scoring.coverage import infer_polarity, score_passage_facets
from app.scoring.calibrate import clamp
from app.scoring.hierarchy import evidence_grade_from_metadata


def _provenance(passage: CandidatePassage) -> dict:
    return {
        "doc_id": passage.doc_id,
        "section": passage.section,
        "offset_start": passage.offset_start,
        "offset_end": passage.offset_end,
        "source_type": passage.source_type.value,
        "guideline_version": passage.metadata.get("guideline_version") or passage.metadata.get("version"),
        "effective_date": passage.metadata.get("effective_date"),
        "timestamp": passage.metadata.get("timestamp"),
    }


def build_retrieval_ledger(
    passages: List[CandidatePassage],
    facets: List[ClinicalFacet],
    agent_id: str = "retrieval",
) -> List[EvidenceLedgerEntry]:
    ledger: List[EvidenceLedgerEntry] = []
    for passage in passages:
        grade = evidence_grade_from_metadata(passage)
        calibrated = clamp(passage.calibrated_score or passage.rerank_score or passage.fusion_score or 0.0)
        facet_scores = score_passage_facets(passage, facets)
        claims: List[EvidenceClaim] = []
        matched = []
        for facet in facets:
            score = facet_scores.get(facet.name, 0.0)
            if score <= 0.0:
                continue
            matched.append(facet.name)
            claims.append(
                EvidenceClaim(
                    facet=facet.name,
                    claim=" ".join(passage.text.split())[:280],
                    polarity=infer_polarity(passage, facet),
                    chunk_id=passage.chunk_id,
                    confidence=clamp(score),
                    evidence_grade=grade,
                    calibrated_relevance=calibrated,
                    agent_id=agent_id,
                    provenance=_provenance(passage),
                )
            )
        ledger.append(
            EvidenceLedgerEntry(
                chunk_id=passage.chunk_id,
                doc_id=passage.doc_id,
                source_type=passage.source_type,
                agent_id=agent_id,
                facets=matched,
                claims=claims,
                calibrated_relevance=calibrated,
                fusion_score=clamp(passage.fusion_score or calibrated),
                evidence_grade=grade,
                safety_relevance=clamp(passage.safety_score or 0.0),
                contradiction_risk=clamp(passage.contradiction_score or 0.0),
                provenance=_provenance(passage),
            )
        )
    return ledger


def merge_agent_claims(
    ledger: List[EvidenceLedgerEntry],
    batches: List[AgentClaimBatch],
    passages: List[CandidatePassage],
) -> List[EvidenceLedgerEntry]:
    by_chunk: Dict[str, EvidenceLedgerEntry] = {entry.chunk_id: entry for entry in ledger}
    passage_map = {p.chunk_id: p for p in passages}
    for batch in batches:
        for claim in batch.claims:
            entry = by_chunk.get(claim.chunk_id)
            if entry is None:
                passage = passage_map.get(claim.chunk_id)
                if not passage:
                    continue
                entry = EvidenceLedgerEntry(
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                    agent_id=batch.agent_id,
                    provenance=_provenance(passage),
                    calibrated_relevance=clamp(passage.calibrated_score or 0.0),
                    fusion_score=clamp(passage.fusion_score or 0.0),
                    evidence_grade=evidence_grade_from_metadata(passage),
                )
                by_chunk[claim.chunk_id] = entry
                ledger.append(entry)
            entry.claims.append(claim)
            if claim.facet not in entry.facets:
                entry.facets.append(claim.facet)
            # Attach agent score onto passage for MAC utility
            passage = passage_map.get(claim.chunk_id)
            if passage is not None:
                passage.agent_scores[batch.agent_id] = max(
                    passage.agent_scores.get(batch.agent_id, 0.0),
                    float(claim.confidence or 0.0),
                )
                if batch.agent_id not in passage.retrieved_by:
                    passage.retrieved_by.append(batch.agent_id)
    return ledger


def detect_contradictions(ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
    by_facet: Dict[str, Dict[str, List[EvidenceLedgerEntry]]] = {}
    for entry in ledger:
        for claim in entry.claims:
            bucket = by_facet.setdefault(claim.facet, {"support": [], "contradict": []})
            if claim.polarity == EvidencePolarity.CONTRADICTS:
                bucket["contradict"].append(entry)
            elif claim.polarity in {
                EvidencePolarity.SUPPORTS,
                EvidencePolarity.QUALIFIES,
                EvidencePolarity.SAFETY,
            }:
                bucket["support"].append(entry)
    pairs: List[ContradictionPair] = []
    for facet, bucket in by_facet.items():
        for support in bucket["support"]:
            for conflict in bucket["contradict"]:
                if support.chunk_id == conflict.chunk_id:
                    continue
                severity = (
                    "high"
                    if support.evidence_grade.score >= 0.80 or conflict.evidence_grade.score >= 0.80
                    else "medium"
                )
                pairs.append(
                    ContradictionPair(
                        facet=facet,
                        chunk_id_a=support.chunk_id,
                        chunk_id_b=conflict.chunk_id,
                        severity=severity,
                        reason="Incompatible support and caution for the same facet.",
                        grade_a=support.evidence_grade.score,
                        grade_b=conflict.evidence_grade.score,
                    )
                )
    return pairs
