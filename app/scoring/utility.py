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
        - settings.CONTRADICTION_PENALTY_WEIGHT * contradiction
        - settings.REDUNDANCY_PENALTY_WEIGHT * _redundancy(candidate, selected)
        - settings.NOISE_PENALTY_WEIGHT * (candidate.noise_score or 0.0)
    )


def _required_source_names(facets: Optional[List[ClinicalFacet]]) -> set:
    if not facets:
        return set()
    return {
        str(facet.source_policy).upper()
        for facet in facets
        if facet.required and facet.source_policy and str(facet.source_policy).upper() in {"LIT", "EMR", "CPG", "SAFETY"}
    }


def _preserve_required_sources(
    selected: List[CandidatePassage],
    remaining: List[CandidatePassage],
    token_budget: int,
    facets: Optional[List[ClinicalFacet]],
) -> None:
    if not facets or not remaining:
        return
    required_sources = _required_source_names(facets)
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
        fitted = _try_fit(
            selected,
            token_budget,
            tokens,
            protected_sources=required_sources,
        )
        if fitted is None:
            continue
        item.selected_reason = f"protected_required_source={source}"
        selected.append(item)
        current_tokens = fitted + tokens


def _is_critical_caution(item: CandidatePassage) -> bool:
    return (item.safety_score or 0.0) >= 0.32 or (item.contradiction_score or 0.0) >= 0.32


def _adds_a_new_drug_warning(item: CandidatePassage, selected: List[CandidatePassage]) -> bool:
    """Keep a caution when it names a drug that no selected caution has named."""
    from app.medswin.ledger import intervention_tokens

    drugs = intervention_tokens(item.text or "")
    warned: set[str] = set()
    unnamed = False
    for chosen in selected:
        if not _is_critical_caution(chosen):
            continue
        named = intervention_tokens(chosen.text or "")
        if named:
            warned |= named
        else:
            unnamed = True
    if drugs:
        return not drugs <= warned
    return not unnamed and not warned


def _repeated_caution(
    selected: List[CandidatePassage],
    protected_sources: Optional[set] = None,
) -> Optional[CandidatePassage]:
    """Return a warning whose drug is already named by another selected warning."""
    from app.medswin.ledger import intervention_tokens

    protected = protected_sources or set()
    victims = []
    for item in selected:
        if not _is_critical_caution(item):
            continue
        drugs = set(intervention_tokens(item.text or ""))
        if not drugs:
            continue
        source = item.source_type.value
        if source in protected and sum(other.source_type.value == source for other in selected) == 1:
            continue
        covered: set = set()
        for other in selected:
            if other is item or not _is_critical_caution(other):
                continue
            covered |= set(intervention_tokens(other.text or ""))
        if drugs <= covered:
            victims.append(item)
    if not victims:
        return None
    return min(victims, key=lambda item: (item.fusion_score or 0.0, item.calibrated_score or 0.0))


def _filler_candidates(
    selected: List[CandidatePassage],
    protected_sources: Optional[set] = None,
) -> List[CandidatePassage]:
    """Return passages that can leave. Keep the last passage of a required source."""
    protected = protected_sources or set()
    fillers = []
    for item in selected:
        if _is_critical_caution(item):
            continue
        source = item.source_type.value
        if source in protected and sum(other.source_type.value == source for other in selected) == 1:
            continue
        fillers.append(item)
    return fillers


def _make_room(
    selected: List[CandidatePassage],
    token_budget: int,
    needed: int,
    protected_sources: Optional[set] = None,
) -> int:
    """Drop the weakest filler so a protected passage can fit. Keep the last passage."""
    current = sum(token_count(item) for item in selected)
    while current + needed > token_budget and len(selected) > 1:
        fillers = _filler_candidates(selected, protected_sources)
        if not fillers:
            break
        victim = min(
            fillers,
            key=lambda item: (item.fusion_score or 0.0, item.calibrated_score or 0.0),
        )
        selected.remove(victim)
        current -= token_count(victim)
    return current


def _try_fit(
    selected: List[CandidatePassage],
    token_budget: int,
    needed: int,
    protected_sources: Optional[set] = None,
) -> Optional[int]:
    """Free a slot for one passage. Restore the bundle when it still does not fit."""
    snapshot = list(selected)
    current = _make_room(selected, token_budget, needed, protected_sources)
    if len(selected) >= settings.MMR_MAX_EVIDENCE_CHUNKS:
        fillers = _filler_candidates(selected, protected_sources)
        if len(selected) <= 1 or not fillers:
            selected[:] = snapshot
            return None
        victim = min(
            fillers,
            key=lambda passage: (passage.fusion_score or 0.0, passage.calibrated_score or 0.0),
        )
        best = max(
            selected,
            key=lambda passage: (passage.fusion_score or 0.0, passage.calibrated_score or 0.0),
        )
        if victim is best:
            selected[:] = snapshot
            return None
        freed = current - token_count(victim)
        if freed + needed > token_budget:
            selected[:] = snapshot
            return None
        selected.remove(victim)
        current = freed
    if current + needed <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
        return current
    selected[:] = snapshot
    return None


def _preserve_safety(
    selected: List[CandidatePassage],
    remaining: List[CandidatePassage],
    token_budget: int,
    protected_sources: Optional[set] = None,
) -> None:
    if not remaining:
        return
    selected_ids = {item.chunk_id for item in selected}
    critical = [
        item for item in remaining
        if item.chunk_id not in selected_ids and _is_critical_caution(item)
    ]
    if not critical:
        return
    current_tokens = sum(token_count(item) for item in selected)
    for item in sorted(critical, key=lambda c: ((c.safety_score or 0.0) + (c.contradiction_score or 0.0), c.fusion_score or 0.0), reverse=True):
        if not _adds_a_new_drug_warning(item, selected):
            continue
        tokens = token_count(item)
        fitted = _try_fit(
            selected,
            token_budget,
            tokens,
            protected_sources=protected_sources,
        )
        if fitted is None:
            victim = _repeated_caution(selected, protected_sources)
            if victim is None:
                continue
            snapshot = list(selected)
            selected.remove(victim)
            current = sum(token_count(passage) for passage in selected)
            if current + tokens <= token_budget and len(selected) < settings.MMR_MAX_EVIDENCE_CHUNKS:
                fitted = current
            else:
                selected[:] = snapshot
                continue
        item.selected_reason = "protected_critical_safety_or_contradiction"
        selected.append(item)
        current_tokens = fitted + tokens


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
    candidates = [
        candidate
        for candidate in candidates
        if candidate.calibrated_score is not None and float(candidate.calibrated_score) >= settings.SUFF_T_INCLUSION
    ]
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
        if selected and best_score <= settings.SUFF_MIN_MARGINAL_UTILITY:
            break
        chosen = remaining.pop(best_idx)
        chosen.selected_reason = f"marginal_utility_per_token={best_score:.6f}"
        used_tokens += token_count(chosen)
        selected.append(chosen)
    _preserve_safety(selected, remaining, token_budget, _required_source_names(facets))
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
