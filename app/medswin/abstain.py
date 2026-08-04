"""Bounded insufficient-evidence response renderer."""

from __future__ import annotations

from typing import List, Optional

from app.schemas.evidence import PolicyDecision
from app.schemas.facets import FacetCoverage
from app.services.medswin.governance import ensure_cds_language, redact_phi_text


def insufficient_answer(
    query: str,
    decision: PolicyDecision,
    coverage: Optional[List[FacetCoverage]] = None,
) -> str:
    """Worked-example style bounded CDS abstention."""
    covered = []
    missing = list(decision.missing_facets)
    for row in coverage or decision.facet_coverage:
        if row.status == "satisfied":
            covered.append(row.facet)
        elif row.status in {"missing", "uncertain", "contradicted"} and row.facet not in missing:
            missing.append(row.facet)

    covered_text = ", ".join(covered) if covered else "none confirmed"
    missing_text = ", ".join(missing) if missing else "required clinical facets"
    conflict = ""
    if decision.unresolved_critical_conflicts:
        conflict = (
            " Unresolved high-severity contradictions were detected between evidence sources; "
            "these are preserved for clinician review rather than averaged away."
        )
    body = (
        f"The available evidence is insufficient to provide a grounded clinician decision-support "
        f"answer for the query: {redact_phi_text(query)}. "
        f"Known/covered facets: {covered_text}. "
        f"Missing or uncertain evidence: {missing_text}.{conflict} "
        "Do not initiate, stop, or change therapy based on this response alone. "
        "Next step: targeted evidence retrieval (e.g., laboratory trend, guideline clause, "
        "or safety monograph) or clinician review of the relevant EMR/guideline source."
    )
    return ensure_cds_language(body)
