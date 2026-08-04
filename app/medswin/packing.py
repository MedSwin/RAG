"""Facet-aware protected pruning of the evidence bundle."""

from __future__ import annotations

from typing import List, Optional

from app.core.config import settings
from app.schemas.evidence import CandidatePassage
from app.schemas.facets import ClinicalFacet
from app.scoring.utility import marginal_utility, token_count


def pack_bundle(
    selected: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
    token_budget: Optional[int] = None,
    epsilon_pack: float = 1e-6,
    safety_crit: float = 0.48,
) -> List[CandidatePassage]:
    """Remove low-utility passages while protecting safety/contradiction evidence."""
    if not selected:
        return selected
    token_budget = token_budget or settings.TOKEN_BUDGET_B
    current = list(selected)
    total = sum(token_count(p) for p in current)
    if total <= token_budget:
        return current

    # Drop lowest utility unprotected passages until under budget
    while total > token_budget and len(current) > 1:
        removable = []
        for idx, passage in enumerate(current):
            safety = passage.safety_score or 0.0
            contradiction = passage.contradiction_score or 0.0
            mu = marginal_utility(passage, [p for i, p in enumerate(current) if i != idx], facets)
            protected = mu > epsilon_pack and (safety >= safety_crit or contradiction >= safety_crit)
            if protected:
                continue
            removable.append((mu, idx))
        if not removable:
            break
        removable.sort(key=lambda item: item[0])
        _, drop_idx = removable[0]
        dropped = current.pop(drop_idx)
        total -= token_count(dropped)
    return current
