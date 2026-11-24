"""Facet alignment and noisy-OR coverage with bootstrap LCB."""

from __future__ import annotations

import math
import random
import re
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


_CAUTION_TOKENS = {"allergy", "adverse", "interaction", "avoid", "toxicity"}


def _token_is_caution(token: str) -> bool:
    if token.startswith("contraindicat"):
        return True
    stems = {token}
    if token.endswith("s") and len(token) > 3:
        stems.add(token[:-1])
    return bool(stems & _CAUTION_TOKENS)


def _safety_score(passage: CandidatePassage) -> float:
    text = passage.text.lower()
    hits = 0
    for match in re.finditer(r"[a-z0-9]+", text):
        if not _token_is_caution(match.group(0)):
            continue
        window = text[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        hits += 1
    if _ungoverned_caution(passage.text):
        hits += 1
    return clamp(hits * 0.18)


def keyword_in_text(keyword: str, text: str) -> bool:
    """Match a keyword as a whole token. A phrase matches as written."""
    needle = (keyword or "").lower().strip()
    if not needle:
        return False
    lowered = (text or "").lower()
    if " " in needle or "-" in needle:
        return needle in lowered
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if needle in tokens:
        return True
    if needle.endswith("s") and needle[:-1] in tokens:
        return True
    return f"{needle}s" in tokens


def score_passage_facets(
    passage: CandidatePassage,
    facets: Iterable[ClinicalFacet],
) -> Dict[str, float]:
    """Score passage-to-facet alignment (paper Eq. facet-alignment)."""
    text = passage.text.lower()
    explicit_scores = passage.facet_scores or passage.metadata.get("facet_scores", {}) or {}
    scores: Dict[str, float] = {}
    for facet in facets:
        source_bonus = _source_policy_score(passage.source_type, facet.source_policy)
        matched = [keyword for keyword in facet.keywords if keyword_in_text(keyword, text)]
        keyword_score = min(0.65, len(matched) * 0.18)
        section_score = passage.section_score if passage.section_score is not None else 0.5
        safety_bonus = 0.25 if facet.name == "safety_contraindications" and _safety_score(passage) > 0 else 0.0
        if not matched and safety_bonus == 0.0 and source_bonus <= 0.45:
            # Source-policy affinity alone must not mark an unrelated facet as covered.
            # An exact source match is 0.45. It still needs a keyword or a safety hit.
            text_score = 0.0
        else:
            text_score = clamp(0.20 + source_bonus + keyword_score + 0.15 * section_score + safety_bonus)
        explicit = explicit_scores.get(facet.name) if isinstance(explicit_scores, dict) else None
        if text_score <= 0.0 or not isinstance(explicit, (int, float)) or isinstance(explicit, bool):
            scores[facet.name] = text_score
        else:
            scores[facet.name] = max(text_score, clamp(explicit))
    passage.facet_scores = scores
    return scores


def _binary_entropy(probability: float) -> float:
    p = clamp(probability, 1e-9, 1.0 - 1e-9)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _contribution(claim: EvidenceClaim, entry: EvidenceLedgerEntry) -> float:
    """Paper coverage term: a_dj * p_hat * g_EBM. No floors that inflate π_j."""
    a_dj = clamp(claim.confidence)
    p_hat = clamp(entry.calibrated_relevance or claim.calibrated_relevance or 0.0)
    g_ebm = clamp(entry.evidence_grade.score if entry.evidence_grade else claim.evidence_grade.score)
    return clamp(a_dj * p_hat * g_ebm)


def _bootstrap_lcb(contributions: Sequence[float], n_boot: int = 64, seed: int = 42) -> float:
    if not contributions:
        return 0.0
    if len(contributions) < 2:
        # One document has no bootstrap sample. Entropy rejects an uncertain probability.
        return clamp(1.0 - math.prod(1.0 - c for c in contributions))
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
        # One contribution per passage, then one per parent document.
        # Heuristic+agent claims on the same chunk must not inflate π_j.
        best_by_chunk: Dict[str, float] = {}
        doc_of_chunk: Dict[str, str] = {}
        for entry in ledger:
            for claim in entry.claims:
                if claim.facet != facet.name:
                    continue
                if claim.polarity == EvidencePolarity.CONTRADICTS:
                    contradicting.append(entry.chunk_id)
                    continue
                contrib = _contribution(claim, entry)
                prev = best_by_chunk.get(entry.chunk_id)
                if prev is None or contrib > prev:
                    best_by_chunk[entry.chunk_id] = contrib
                    doc_of_chunk[entry.chunk_id] = entry.doc_id or entry.chunk_id
                if claim.polarity != EvidencePolarity.IRRELEVANT:
                    supporting.append(entry.chunk_id)
        best_by_doc: Dict[str, float] = {}
        for chunk_id, contrib in best_by_chunk.items():
            doc_id = doc_of_chunk.get(chunk_id, chunk_id)
            prev = best_by_doc.get(doc_id)
            if prev is None or contrib > prev:
                best_by_doc[doc_id] = contrib
        contributions = list(best_by_doc.values())
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


_CAUTION = re.compile(r"\b(?:contraindicat\w*|not recommended|\bavoid\b|do not|should not)\b", re.I)
_LOCAL_NEGATION = re.compile(
    r"\b(?:no|not|without|absent|denies|denied|negative for|free of)\b(?:\W+\w+){0,4}\W*$",
    re.I,
)
_BENIGN_DO_NOT = re.compile(r"^\s+(?:delay|withhold|wait|hesitate)\b", re.I)
_QUALIFY = re.compile(r"\b(?:unless|except|cautions?|precautions?|monitor)\b", re.I)


def _ungoverned_caution(text: str) -> bool:
    """True when caution wording is not cancelled by a nearby negation."""
    for match in _CAUTION.finditer(text or ""):
        window = text[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        phrase = match.group(0).lower()
        if phrase in {"do not", "should not"} and _BENIGN_DO_NOT.match(text[match.end():]):
            continue
        return True
    return False


def infer_polarity(passage: CandidatePassage, facet: ClinicalFacet) -> EvidencePolarity:
    explicit = passage.metadata.get("polarity")
    if explicit:
        try:
            return EvidencePolarity(explicit)
        except ValueError:
            pass
    text = passage.text.lower()
    if _ungoverned_caution(text):
        return EvidencePolarity.CONTRADICTS if facet.name != "safety_contraindications" else EvidencePolarity.SAFETY
    if _QUALIFY.search(text):
        return EvidencePolarity.QUALIFIES
    return EvidencePolarity.SUPPORTS
