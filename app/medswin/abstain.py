"""Bounded insufficient-evidence response renderer."""

from __future__ import annotations

from typing import Any, List, Optional

from app.schemas.evidence import PolicyDecision
from app.schemas.facets import FacetCoverage
from app.services.medswin.governance import ensure_cds_language, redact_phi_text


_ABSTAIN_MARKERS = (
    "insufficient",
    "no evidence",
    "not contain",
    "do not contain",
    "cannot answer",
    "cannot be answered",
    "unavailable",
    "unanswered",
    "lack of evidence",
)


def citations_match(
    valid_ids: list[str],
    hallucinated_ids: list[str],
    expect_abstain: bool,
) -> bool:
    """An answerable case needs one real citation. A made-up citation fails either case."""
    if hallucinated_ids:
        return False
    if expect_abstain:
        return True
    return len(valid_ids) >= 1


def abstention_matches(content: str, expect_abstain: bool) -> bool:
    """Return whether a required abstention is present. An answerable case stays open."""
    if not expect_abstain:
        return True
    lowered = (content or "").lower()
    return any(marker in lowered for marker in _ABSTAIN_MARKERS)


def insufficient_answer(
    query: str,
    decision: PolicyDecision,
    coverage: Optional[List[FacetCoverage]] = None,
    passages: Optional[List[Any]] = None,
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
        from app.medswin.cds import _caution_drug_name

        drug = _caution_drug_name(
            passages or [], decision, decision.contradictions, severity="high"
        )
        named = f" for {drug}" if drug else ""
        conflict = (
            f" Unresolved high-severity contradictions{named} were detected between evidence sources; "
            "these are preserved for clinician review rather than averaged away."
        )
    from app.medswin.cds import _caution_drug_name, _caution_summary, _conflict_summary

    if any(item.severity == "medium" for item in decision.contradictions if not item.resolved):
        conflict += (
            f" {_conflict_summary(_caution_drug_name(passages or [], decision, decision.contradictions, severity='medium'))}"
            " Review the paired evidence before a therapy change."
        )
    if any(item.severity == "low" for item in decision.contradictions if not item.resolved):
        caution = _caution_summary(
            _caution_drug_name(passages or [], decision, decision.contradictions)
        )
        conflict += f" {caution} Review the paired evidence before a therapy change."
    from app.medswin.cds import _chart_caution_line

    chart_line = _chart_caution_line(decision, decision.contradictions)
    if chart_line and chart_line.lower() not in conflict.lower():
        sentence = chart_line if chart_line.endswith(".") else f"{chart_line}."
        conflict += f" {sentence}"
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
