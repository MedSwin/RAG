"""Facet alignment and noisy-OR coverage with bootstrap LCB."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

from app.core.config import settings
from app.schemas.enums import EvidencePolarity, SourceType
from app.schemas.evidence import CandidatePassage, EvidenceClaim, EvidenceLedgerEntry
from app.schemas.facets import ClinicalFacet, FacetCoverage
from app.scoring.calibrate import clamp
from app.scoring.hierarchy import evidence_grade_from_metadata


def _source_policy_score(source_type: SourceType, source_policy: Optional[str]) -> float:
    if not source_policy:
        return 0.0
    if source_policy == "ANY":
        return 0.20
    if source_policy == source_type.value:
        return 0.45
    if source_policy == "CPG" and source_type in {SourceType.LIT, SourceType.SAFETY}:
        return 0.15
    if source_policy == "SAFETY" and source_type in {SourceType.CPG, SourceType.LIT}:
        return 0.20
    return 0.0


def _safety_score(passage: CandidatePassage) -> float:
    text = passage.text.lower()
    terms = ["contraindicat", "allergy", "adverse", "interaction", "avoid", "risk", "toxicity", "dose"]
    return clamp(sum(1 for term in terms if term in text) * 0.18)


def score_passage_facets(
    passage: CandidatePassage,
    facets: Iterable[ClinicalFacet],
) -> Dict[str, float]:
    """Score passage-to-facet alignment (paper Eq. facet-alignment)."""
    text = passage.text.lower()
    explicit_scores = passage.facet_scores or passage.metadata.get("facet_scores", {})
    scores: Dict[str, float] = {}
    for facet in facets:
        if facet.name in explicit_scores:
            scores[facet.name] = clamp(explicit_scores[facet.name])
            continue
        source_bonus = _source_policy_score(passage.source_type, facet.source_policy)
        matched = [keyword for keyword in facet.keywords if keyword.lower() in text]
        keyword_score = min(0.65, len(matched) * 0.18)
        section_score = passage.section_score if passage.section_score is not None else 0.5
        safety_bonus = 0.25 if facet.name == "safety_contraindications" and _safety_score(passage) > 0 else 0.0
        if source_bonus == 0.0 and not matched and safety_bonus == 0.0:
            scores[facet.name] = 0.0
        else:
            scores[facet.name] = clamp(0.20 + source_bonus + keyword_score + 0.15 * section_score + safety_bonus)
    passage.facet_scores = scores
    return scores


def _binary_entropy(probability: float) -> float:
    p = clamp(probability, 1e-9, 1.0 - 1e-9)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _contribution(claim: EvidenceClaim, entry: EvidenceLedgerEntry) -> float:
    """Paper coverage term: a_dj * p_hat * g_EBM."""
    a_dj = clamp(claim.confidence)
    p_hat = clamp(entry.calibrated_relevance or claim.calibrated_relevance or 0.0)
    g_ebm = clamp(entry.evidence_grade.score if entry.evidence_grade else claim.evidence_grade.score)
    contribution = clamp(a_dj * max(p_hat, 0.35) * max(g_ebm, 0.35))
    if claim.polarity == EvidencePolarity.CONTRADICTS:
        contribution *= 0.35
    return contribution


def _bootstrap_lcb(contributions: Sequence[float], n_boot: int = 64, seed: int = 42) -> float:
    if not contributions:
        return 0.0
    if len(contributions) < 2:
        prob = clamp(1.0 - math.prod(1.0 - c for c in contributions))
        return clamp(prob - settings.SUFF_LCB_MARGIN)
    rng = random.Random(seed)
    samples = []
    n = len(contributions)
    for _ in range(n_boot):
        draw = [contributions[rng.randrange(n)] for _ in range(n)]
        no_support = 1.0
        for c in draw:
            no_support *= 1.0 - clamp(c)
        samples.append(clamp(1.0 - no_support))
    samples.sort()
    idx = max(0, int(0.05 * (len(samples) - 1)))
    return samples[idx]


def compute_facet_coverage(
    facets: List[ClinicalFacet],
    ledger: List[EvidenceLedgerEntry],
) -> List[FacetCoverage]:
    """Compute noisy-OR coverage with bootstrap LCB."""
    coverage: List[FacetCoverage] = []
    for facet in facets:
        contributions: List[float] = []
        supporting: List[str] = []
        contradicting: List[str] = []
        for entry in ledger:
            for claim in entry.claims:
                if claim.facet != facet.name:
                    continue
                contrib = _contribution(claim, entry)
                contributions.append(contrib)
                if claim.polarity == EvidencePolarity.CONTRADICTS:
                    contradicting.append(entry.chunk_id)
                else:
                    supporting.append(entry.chunk_id)
        no_support = 1.0
        for c in contributions:
            no_support *= 1.0 - clamp(c)
        probability = clamp(1.0 - no_support)
        lcb = _bootstrap_lcb(contributions)
        entropy = _binary_entropy(probability)
        if lcb >= facet.threshold and entropy <= settings.SUFF_MAX_ENTROPY:
            status = "satisfied"
        elif supporting:
            status = "uncertain"
        elif contradicting:
            status = "contradicted"
        else:
            status = "missing"
        coverage.append(
            FacetCoverage(
                facet=facet.name,
                required=facet.required,
                threshold=facet.threshold,
                coverage_probability=probability,
                lower_confidence_bound=lcb,
                entropy=entropy,
                status=status,
                supporting_chunk_ids=sorted(set(supporting)),
                contradicting_chunk_ids=sorted(set(contradicting)),
            )
        )
    return coverage


def infer_polarity(passage: CandidatePassage, facet: ClinicalFacet) -> EvidencePolarity:
    explicit = passage.metadata.get("polarity")
    if explicit:
        try:
            return EvidencePolarity(explicit)
        except ValueError:
            pass
    text = passage.text.lower()
    if any(term in text for term in ["contraindicat", "not recommended", "avoid", "do not", "should not"]):
        return EvidencePolarity.CONTRADICTS if facet.name != "safety_contraindications" else EvidencePolarity.SAFETY
    if any(term in text for term in ["unless", "except", "caution", "monitor"]):
        return EvidencePolarity.QUALIFIES
    return EvidencePolarity.SUPPORTS
