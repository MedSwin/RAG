"""Claim-level evidence ledger merge."""

from __future__ import annotations

import re
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
from app.scoring.calibrate import clamp, first_present
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
        scored = first_present(passage.calibrated_score, passage.rerank_score)
        if scored is None:
            # A rerank miss must not wipe facet coverage. Inclusion still requires calibrated_score.
            scored = first_present(passage.fusion_score, passage.dense_score, default=0.0)
        calibrated = clamp(scored)
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
                fusion_score=clamp(first_present(passage.fusion_score, default=calibrated)),
                evidence_grade=grade,
                safety_relevance=clamp(passage.safety_score or 0.0),
                contradiction_risk=clamp(passage.contradiction_score or 0.0),
                text=" ".join(passage.text.split()),
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
            passage = passage_map.get(claim.chunk_id)
            if passage is None:
                continue
            text_score = float((passage.facet_scores or {}).get(claim.facet, 0.0) or 0.0)
            if text_score <= 0.0:
                continue
            text_polarity = infer_polarity(passage, ClinicalFacet(name=claim.facet))
            if (
                text_polarity
                in {
                    EvidencePolarity.CONTRADICTS,
                    EvidencePolarity.SAFETY,
                    EvidencePolarity.QUALIFIES,
                }
                and claim.polarity == EvidencePolarity.SUPPORTS
            ):
                claim = claim.model_copy(update={"polarity": text_polarity})
            entry = by_chunk.get(claim.chunk_id)
            if entry is None:
                entry = EvidenceLedgerEntry(
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                    agent_id=batch.agent_id,
                    provenance=_provenance(passage),
                    calibrated_relevance=clamp(passage.calibrated_score or 0.0),
                    fusion_score=clamp(passage.fusion_score or 0.0),
                    evidence_grade=evidence_grade_from_metadata(passage),
                    text=" ".join(passage.text.split()),
                )
                by_chunk[claim.chunk_id] = entry
                ledger.append(entry)
            entry.claims.append(claim)
            if claim.facet not in entry.facets:
                entry.facets.append(claim.facet)
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
                if current <= 0.0:
                    continue
                passage.facet_scores[claim.facet] = max(current, clamp(claim.confidence))


_STALE_TERMS = ("outdated", "obsolete", "superseded", "population mismatch")


def adjudicate_contradictions(
    pairs: List[ContradictionPair],
    batches: List[AgentClaimBatch],
) -> List[ContradictionPair]:
    """Resolve one medium conflict only when the critic cites both chunks."""
    cited_stale: set[str] = set()
    for batch in batches:
        if batch.agent_id != "critic":
            continue
        for claim in batch.claims:
            text = f"{claim.claim} {claim.chunk_id}".lower()
            if claim.chunk_id and any(term in text for term in _STALE_TERMS):
                cited_stale.add(claim.chunk_id)
    for pair in pairs:
        if pair.resolved or pair.severity == "high":
            continue
        if pair.chunk_id_a in cited_stale and pair.chunk_id_b in cited_stale:
            pair.resolved = True
            pair.adjudication = "critic cited both chunks as outdated or mismatched"
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


_DRUG_WORDS = {
    "apixaban",
    "dabigatran",
    "rivaroxaban",
    "edoxaban",
    "warfarin",
    "heparin",
    "enoxaparin",
    "aspirin",
    "nsaid",
    "nsaids",
    "ibuprofen",
    "ketorolac",
    "metformin",
}
_DRUG_SUFFIXES = (
    "xaban",
    "gatran",
    "farin",
    "parin",
    "profen",
    "cillin",
    "mycin",
    "statin",
    "sartan",
    "pril",
    "olol",
    "azole",
    "mab",
    "nib",
)
_WORD = re.compile(r"[a-z][a-z0-9-]{3,}")
_NOT_DRUGS = {"april"}


def intervention_tokens(text: str) -> set[str]:
    """Return named drugs in one passage. Class words such as anticoagulant stay out."""
    found: set[str] = set()
    for raw in _WORD.findall((text or "").lower()):
        for word in raw.split("-"):
            if len(word) < 4 or word in _NOT_DRUGS:
                continue
            if word.endswith("s") and word[:-1] in _DRUG_WORDS:
                word = word[:-1]
            listed = next((drug for drug in _DRUG_WORDS if len(drug) >= 6 and word.startswith(drug)), None)
            if listed:
                found.add(listed)
            elif word in _DRUG_WORDS or word.endswith(_DRUG_SUFFIXES):
                found.add(word)
    return found


def same_intervention(left: str, right: str) -> bool:
    """Pair a caution with support only when both name the same drug, or neither names one.

    A named-drug warning does not contradict a passage that names a different drug
    or no drug. The warning still covers the safety facet.
    """
    left_drugs = intervention_tokens(left)
    right_drugs = intervention_tokens(right)
    if left_drugs and right_drugs:
        return not left_drugs.isdisjoint(right_drugs)
    if left_drugs or right_drugs:
        return False
    return True


def _pair_severity(left: EvidenceLedgerEntry, right: EvidenceLedgerEntry) -> str:
    return (
        "high"
        if left.evidence_grade.score >= 0.80 or right.evidence_grade.score >= 0.80
        else "medium"
    )


def _pairing_text(entry: EvidenceLedgerEntry, claim: Optional[EvidenceClaim] = None) -> str:
    """Use the full passage when the claim preview was cut."""
    stored = (entry.text or "").strip()
    if stored:
        return stored
    if claim is not None and claim.claim:
        return claim.claim
    return " ".join(item.claim for item in entry.claims)


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

    def _add(
        facet: str,
        support: EvidenceLedgerEntry,
        conflict: EvidenceLedgerEntry,
        reason: str,
        severity: str | None = None,
    ) -> None:
        key = (facet, support.chunk_id, conflict.chunk_id)
        if support.chunk_id == conflict.chunk_id or key in seen:
            return
        seen.add(key)
        pairs.append(
            ContradictionPair(
                facet=facet,
                chunk_id_a=support.chunk_id,
                chunk_id_b=conflict.chunk_id,
                severity=severity or _pair_severity(support, conflict),
                reason=reason,
                grade_a=support.evidence_grade.score,
                grade_b=conflict.evidence_grade.score,
            )
        )

    for facet, bucket in by_facet.items():
        for support in bucket["support"]:
            for conflict in bucket["contradict"]:
                support_text = _pairing_text(support)
                conflict_text = _pairing_text(conflict)
                if not same_intervention(support_text, conflict_text):
                    continue
                _add(facet, support, conflict, "Incompatible support and caution for the same intervention.")

    safety_warnings: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    other_supports: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    for entry in ledger:
        for claim in entry.claims:
            if claim.facet == "safety_contraindications" and claim.polarity in {
                EvidencePolarity.SAFETY,
                EvidencePolarity.CONTRADICTS,
            }:
                safety_warnings.append((entry, claim))
            elif (
                claim.polarity == EvidencePolarity.SUPPORTS
                and claim.facet != "safety_contraindications"
            ):
                other_supports.append((entry, claim))
    for support_entry, support_claim in other_supports:
        for warning_entry, warning_claim in safety_warnings:
            # A safety warning crosses facets only when both sides name the same drug.
            # An unnamed caution still covers the safety facet. It does not veto every guideline.
            if not intervention_tokens(_pairing_text(support_entry, support_claim)) or not intervention_tokens(
                _pairing_text(warning_entry, warning_claim)
            ):
                continue
            if not same_intervention(
                _pairing_text(support_entry, support_claim),
                _pairing_text(warning_entry, warning_claim),
            ):
                continue
            _add(
                support_claim.facet,
                support_entry,
                warning_entry,
                "Same-drug safety warning conflicts with this recommendation.",
            )
    qualifies: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    plain_supports: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    for entry in ledger:
        for claim in entry.claims:
            if claim.polarity == EvidencePolarity.QUALIFIES:
                qualifies.append((entry, claim))
            elif claim.polarity == EvidencePolarity.SUPPORTS:
                plain_supports.append((entry, claim))
    for support_entry, support_claim in plain_supports:
        for qualify_entry, qualify_claim in qualifies:
            if not intervention_tokens(_pairing_text(support_entry, support_claim)) or not intervention_tokens(
                _pairing_text(qualify_entry, qualify_claim)
            ):
                continue
            if not same_intervention(
                _pairing_text(support_entry, support_claim),
                _pairing_text(qualify_entry, qualify_claim),
            ):
                continue
            _add(
                support_claim.facet,
                support_entry,
                qualify_entry,
                "Same-drug caution qualifies this recommendation.",
                severity="low",
            )
    return pairs
