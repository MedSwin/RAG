"""Apply retrieve-more hints to query/constraints/K."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.enums import SourceType


_FACET_QUERY_STOP = {"safety", "patient", "evidence", "quality"}


SYNONYM_MAP = {
    "metformin": ["biguanide", "glucophage"],
    "renal": ["kidney", "egfr", "creatinine", "ckd"],
    "contraindication": ["contraindicated", "avoid"],
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
    missing = hints.get("missing_facets") or []
    current = query
    seen: set[str] = set()
    for _ in range(8):
        lower = current.lower()
        extras: List[str] = []
        for term, syns in SYNONYM_MAP.items():
            if term not in lower:
                continue
            for syn in syns:
                key = syn.lower()
                if key in lower or key in seen:
                    continue
                seen.add(key)
                extras.append(syn)
        for facet in missing:
            for word in str(facet).replace("_", " ").split():
                key = word.lower()
                if key in lower or key in seen or key in _FACET_QUERY_STOP:
                    continue
                seen.add(key)
                extras.append(word)
        if not extras:
            return current
        addition = " ".join(sorted(set(extras)))
        marker = "patient retrieval context:"
        index = current.lower().find(marker)
        if index >= 0:
            current = f"{current[:index].rstrip()} {addition}\n\n{current[index:]}"
        else:
            current = f"{current} {addition}"
    return current


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
    # Exclusive source focus only when a single source is requested.
    # Multiple missing facets must not starve CPG while searching SAFETY (or vice versa).
    focus = hints.get("focus_source")
    exclusive = bool(hints.get("exclusive_source", True))
    if focus and exclusive:
        try:
            source_filter = SourceType(str(focus).upper())
            constraints["source_policy"] = f"{source_filter.value}_ONLY"
        except ValueError:
            pass
    elif focus and not exclusive:
        constraints["preferred_source"] = str(focus).upper()

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
