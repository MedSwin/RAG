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


def filter_ledger(
    ledger: List[EvidenceLedgerEntry],
    chunk_ids: set,
) -> List[EvidenceLedgerEntry]:
    """Keep merged agent+retrieval claims for the selected bundle only."""
    return [entry.model_copy(deep=True) for entry in ledger if entry.chunk_id in chunk_ids]


def apply_claim_alignment(
    passages: List[CandidatePassage],
    batches: List[AgentClaimBatch],
    facets: List[ClinicalFacet],
) -> None:
    """Lift specialist claim confidence into passage facet scores for utility."""
    names = {facet.name for facet in facets}
    for batch in batches:
        for claim in batch.claims:
            if claim.facet not in names:
                continue
            for passage in passages:
                if passage.chunk_id != claim.chunk_id:
                    continue
                current = passage.facet_scores.get(claim.facet, 0.0)
                passage.facet_scores[claim.facet] = max(current, clamp(claim.confidence))


def adjudicate_contradictions(
    pairs: List[ContradictionPair],
    batches: List[AgentClaimBatch],
) -> List[ContradictionPair]:
    """Resolve low/medium conflicts when the critic cites outdated or mismatched evidence."""
    notes = " ".join(
        " ".join(batch.notes) for batch in batches if batch.agent_id == "critic"
    ).lower()
    critic_text = notes + " " + " ".join(
        claim.claim.lower()
        for batch in batches if batch.agent_id == "critic"
        for claim in batch.claims
    )
    stale = any(term in critic_text for term in ("outdated", "obsolete", "superseded", "population mismatch"))
    for pair in pairs:
        if pair.resolved or pair.severity == "high":
            continue
        if stale or abs(pair.grade_a - pair.grade_b) >= 0.25:
            pair.resolved = True
            pair.adjudication = "critic: outdated, population mismatch, or evidence-grade dominance"
    return pairs


def _contradicts_facet(claim: EvidenceClaim, facet: str) -> bool:
    if claim.polarity == EvidencePolarity.CONTRADICTS:
        return True
    # A safety warning on a treatment/guideline facet is a conflict, not support.
    return claim.polarity == EvidencePolarity.SAFETY and facet != "safety_contraindications"


def _supports_facet(claim: EvidenceClaim, facet: str) -> bool:
    if claim.polarity == EvidencePolarity.IRRELEVANT:
        return False
    if _contradicts_facet(claim, facet):
        return False
    return claim.polarity in {
        EvidencePolarity.SUPPORTS,
        EvidencePolarity.QUALIFIES,
        EvidencePolarity.SAFETY,
    }


def _pair_severity(left: EvidenceLedgerEntry, right: EvidenceLedgerEntry) -> str:
    return (
        "high"
        if left.evidence_grade.score >= 0.80 or right.evidence_grade.score >= 0.80
        else "medium"
    )


def detect_contradictions(ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
    by_facet: Dict[str, Dict[str, List[EvidenceLedgerEntry]]] = {}
    for entry in ledger:
        for claim in entry.claims:
            bucket = by_facet.setdefault(claim.facet, {"support": [], "contradict": []})
            if _contradicts_facet(claim, claim.facet):
                bucket["contradict"].append(entry)
            elif _supports_facet(claim, claim.facet):
                bucket["support"].append(entry)
    pairs: List[ContradictionPair] = []
    seen = set()

    def _add(facet: str, support: EvidenceLedgerEntry, conflict: EvidenceLedgerEntry, reason: str) -> None:
        key = (facet, support.chunk_id, conflict.chunk_id)
        if support.chunk_id == conflict.chunk_id or key in seen:
            return
        seen.add(key)
        pairs.append(
            ContradictionPair(
                facet=facet,
                chunk_id_a=support.chunk_id,
                chunk_id_b=conflict.chunk_id,
                severity=_pair_severity(support, conflict),
                reason=reason,
                grade_a=support.evidence_grade.score,
                grade_b=conflict.evidence_grade.score,
            )
        )

    for facet, bucket in by_facet.items():
        for support in bucket["support"]:
            for conflict in bucket["contradict"]:
                _add(facet, support, conflict, "Incompatible support and caution for the same facet.")
    return pairs
