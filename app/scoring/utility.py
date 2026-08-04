"""Budgeted greedy evidence-bundle selection."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import tiktoken
except ModuleNotFoundError:
    class _WhitespaceEncoding:
        def encode(self, text):
            return text.split()

    class tiktoken:
        @staticmethod
        def get_encoding(_name):
            return _WhitespaceEncoding()

from app.core.config import settings
from app.schemas.evidence import CandidatePassage
from app.schemas.facets import ClinicalFacet
from app.scoring.calibrate import clamp
from app.scoring.hierarchy import evidence_grade_from_metadata

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def token_count(candidate: CandidatePassage) -> int:
    if candidate.token_count:
        return candidate.token_count
    candidate.token_count = len(_TOKENIZER.encode(candidate.text))
    return candidate.token_count


def _redundancy(candidate: CandidatePassage, selected: List[CandidatePassage]) -> float:
    if not selected:
        return 0.0
    terms = set(candidate.text.lower().split())
    if not terms:
        return 0.0
    max_overlap = 0.0
    for passage in selected:
        other = set(passage.text.lower().split())
        if not other:
            continue
        overlap = len(terms & other) / len(terms | other)
        same_doc = 0.15 if passage.doc_id == candidate.doc_id else 0.0
        max_overlap = max(max_overlap, overlap + same_doc)
    return clamp(max_overlap)


def _facet_gain(
    candidate: CandidatePassage,
    selected: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]],
) -> float:
    if not facets:
        return 0.0
    covered = set()
    for passage in selected:
        covered.update(name for name, score in passage.facet_scores.items() if score > 0.25)
    gain = 0.0
    for facet in facets:
        score = candidate.facet_scores.get(facet.name, 0.0)
        if score <= 0.0:
            continue
        novelty = 1.0 if facet.name not in covered else 0.35
        gain += facet.weight * score * novelty
    return gain


def marginal_utility(
    candidate: CandidatePassage,
    selected: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]],
    agent_weights: Optional[Dict[str, float]] = None,
) -> float:
    relevance = candidate.fusion_score or 0.0
    if agent_weights and candidate.agent_scores:
        weighted = sum(agent_weights.get(aid, 0.0) * score for aid, score in candidate.agent_scores.items())
        relevance = 0.7 * relevance + 0.3 * weighted
    safety = candidate.safety_score or 0.0
    contradiction = candidate.contradiction_score or 0.0
    ebm = candidate.evidence_grade_score or evidence_grade_from_metadata(candidate).score
    return (
        relevance
        + settings.SAFETY_REWARD_WEIGHT * safety
        + 0.25 * ebm
        + _facet_gain(candidate, selected, facets)
        + 0.12 * contradiction
        - settings.REDUNDANCY_PENALTY_WEIGHT * _redundancy(candidate, selected)
        - settings.NOISE_PENALTY_WEIGHT * (candidate.noise_score or 0.0)
    )


def _preserve_required_sources(
    selected: List[CandidatePassage],
    remaining: List[CandidatePassage],
    token_budget: int,
    facets: Optional[List[ClinicalFacet]],
) -> None:
    if not facets or not remaining:
        return
    required_sources = {
        str(facet.source_policy).upper()
        for facet in facets
        if facet.required and facet.source_policy and str(facet.source_policy).upper() in {"LIT", "EMR", "CPG", "SAFETY"}
    }
    if not required_sources:
        return
    selected_sources = {item.source_type.value for item in selected}
    current_tokens = sum(token_count(item) for item in selected)
    for source in sorted(required_sources - selected_sources):
        candidates = [item for item in remaining if item.source_type.value == source]
        if not candidates:
            continue
        item = max(candidates, key=lambda c: (max((c.facet_scores or {}).values(), default=0.0), c.fusion_score or 0.0))
        tokens = token_count(item)
        if current_tokens + tokens <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
            item.selected_reason = f"protected_required_source={source}"
            selected.append(item)
            current_tokens += tokens


def _preserve_safety(
    selected: List[CandidatePassage],
    remaining: List[CandidatePassage],
    token_budget: int,
) -> None:
    if not remaining:
        return
    selected_ids = {item.chunk_id for item in selected}
    critical = [
        item for item in remaining
        if item.chunk_id not in selected_ids
        and ((item.safety_score or 0.0) >= 0.48 or (item.contradiction_score or 0.0) >= 0.48)
    ]
    if not critical:
        return
    current_tokens = sum(token_count(item) for item in selected)
    for item in sorted(critical, key=lambda c: ((c.safety_score or 0.0) + (c.contradiction_score or 0.0), c.fusion_score or 0.0), reverse=True):
        tokens = token_count(item)
        if current_tokens + tokens <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
            item.selected_reason = "protected_critical_safety_or_contradiction"
            selected.append(item)
            return


def select_bundle(
    candidates: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
    max_chunks: Optional[int] = None,
    token_budget: Optional[int] = None,
    agent_weights: Optional[Dict[str, float]] = None,
) -> List[CandidatePassage]:
    """Greedy max marginal utility per token under budget."""
    if not candidates:
        return []
    max_chunks = max_chunks or settings.MMR_MAX_EVIDENCE_CHUNKS
    token_budget = token_budget or settings.TOKEN_BUDGET_B
    selected: List[CandidatePassage] = []
    remaining = sorted(candidates.copy(), key=lambda item: item.fusion_score or 0.0, reverse=True)
    used_tokens = 0
    while remaining and len(selected) < max_chunks:
        best_score = -float("inf")
        best_idx = -1
        for idx, candidate in enumerate(remaining):
            tokens = token_count(candidate)
            if used_tokens + tokens > token_budget:
                continue
            score = marginal_utility(candidate, selected, facets, agent_weights) / max(tokens, 1)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        chosen.selected_reason = f"marginal_utility_per_token={best_score:.6f}"
        used_tokens += token_count(chosen)
        selected.append(chosen)
    _preserve_safety(selected, remaining, token_budget)
    _preserve_required_sources(selected, remaining, token_budget, facets)
    return selected


def estimate_marginal_utility_per_token(
    passages: List[CandidatePassage],
    coverage,
) -> float:
    if not passages:
        return 1.0
    deficit = sum(max(0.0, item.threshold - item.lower_confidence_bound) for item in coverage if item.required)
    total_tokens = sum(token_count(p) for p in passages)
    return deficit / max(total_tokens, 1)
