"""Apply retrieve-more hints to query/constraints/K."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.enums import SourceType


SYNONYM_MAP = {
    "metformin": ["biguanide", "glucophage"],
    "renal": ["kidney", "egfr", "creatinine", "ckd"],
    "contraindication": ["contraindicated", "do not use", "avoid", "not recommended"],
    "allergy": ["hypersensitivity", "allergic reaction"],
    "hypertension": ["high blood pressure", "htn"],
    "diabetes": ["dm", "t2dm", "type 2 diabetes"],
}


def resolve_k(hints: Optional[Dict[str, Any]], top_k: Optional[int] = None) -> int:
    k = top_k or settings.CANDIDATE_K
    if hints and hints.get("increase_k"):
        return settings.CANDIDATE_K_PRIME
    return k


def expand_query(query: str, hints: Optional[Dict[str, Any]]) -> str:
    if not hints or not hints.get("expand_synonyms"):
        return query
    extras: List[str] = []
    lower = query.lower()
    for term, syns in SYNONYM_MAP.items():
        if term in lower:
            extras.extend(syns)
    missing = hints.get("missing_facets") or []
    for facet in missing:
        extras.extend(str(facet).replace("_", " ").split())
    if not extras:
        return query
    return f"{query} {' '.join(sorted(set(extras)))}"


def apply_hints(
    query: str,
    constraints: Optional[Dict[str, Any]],
    hints: Optional[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], int, Optional[SourceType]]:
    """Return expanded query, mutated constraints, k, and optional source focus."""
    constraints = dict(constraints or {})
    hints = hints or {}
    k = resolve_k(hints, top_k)
    expanded = expand_query(query, hints)

    source_filter: Optional[SourceType] = None
    focus = hints.get("focus_source")
    if focus:
        try:
            source_filter = SourceType(str(focus).upper())
            constraints["source_policy"] = f"{source_filter.value}_ONLY"
        except ValueError:
            pass

    if hints.get("safety_search"):
        constraints["safety_search"] = True
        if not source_filter:
            # Prefer SAFETY corpus but do not hard-exclude others unless focused.
            constraints.setdefault("source_policy", constraints.get("source_policy") or "ANY")

    if hints.get("relax_filters"):
        constraints.pop("min_evidence_grade", None)
        constraints.pop("timeframe", None)
        constraints["relaxed"] = True

    if hints.get("contradiction_review"):
        constraints["contradiction_review"] = True
        k = max(k, settings.CANDIDATE_K_PRIME)

    return expanded, constraints, k, source_filter
