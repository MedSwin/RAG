"""CDS Hooks 2.0.1 cards from an existing MedSwin decision."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Sequence

from app.schemas.enums import PolicyAction
from app.schemas.traces import ChatResponse
from app.services.medswin.governance import redact_phi_text

SERVICE_ID = "medswin-evidence"
SOURCE_LABEL = "MedSwin"
_SUMMARY_LIMIT = 140


def discovery() -> Dict[str, Any]:
    """Return the CDS Hooks discovery document."""
    return {
        "services": [
            {
                "hook": "patient-view",
                "id": SERVICE_ID,
                "title": "MedSwin evidence check",
                "description": (
                    "Returns one card from the two-stage evidence gate. "
                    "Send the clinical question in context.question. "
                    "A request with no question returns one warning card. "
                    "POST /cds-services/medswin-evidence/feedback records an accepted or overridden card."
                ),
            }
        ]
    }


def missing_question_card(patient_id: str | None = None) -> Dict[str, Any]:
    """Return one warning card when patient-view has no clinical question."""
    who = (patient_id or "").strip() or "this patient"
    return {
        "uuid": str(uuid.uuid4()),
        "summary": "No clinical question was sent for this patient view.",
        "indicator": "warning",
        "source": {"label": SOURCE_LABEL},
        "detail": (
            f"The patient-view hook for {who} did not include a clinical question. "
            "MedSwin did not run the evidence gate."
        ),
    }


def cards_from_chat(
    response: ChatResponse,
    *,
    hook_instance: str | None = None,
    patient_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Map one chat decision to zero or one CDS card."""
    decision = response.policy_decision
    passages = list(response.evidence_bundle.passages or [])
    passage_ids = [passage.chunk_id for passage in passages if passage.chunk_id]
    missing = list(decision.missing_facets) if decision else []
    contradictions = list(response.contradictions or [])
    unresolved = bool(decision and decision.unresolved_critical_conflicts)
    abstained = bool(response.answer_artifact and response.answer_artifact.abstained)
    crashed = bool((response.degraded_mode or {}).get("error")) or decision is None
    synthesis_failed = bool((response.degraded_mode or {}).get("synthesis")) or bool(
        decision and decision.passed and abstained
    )
    accepted = bool(
        decision
        and not crashed
        and decision.passed
        and decision.action == PolicyAction.ACCEPT
        and passages
        and not abstained
        and not unresolved
    )
    if crashed:
        indicator = "critical"
        summary = "MedSwin could not complete the evidence check."
    elif accepted:
        indicator = "info"
        summary = _clinical_summary(response.answer)
    elif unresolved:
        indicator = "critical"
        conflict_drug = _caution_drug_name(passages, decision, contradictions, severity="high")
        summary = (
            _named_summary(
                conflict_drug,
                "Unresolved evidence conflicts for ",
                " need clinician review.",
                "Conflicts for ",
            )
            if conflict_drug
            else "Unresolved evidence conflicts need clinician review."
        )
    elif synthesis_failed:
        indicator = "warning"
        summary = "The model did not return a structured clinical answer."
    else:
        indicator = "warning"
        parts = []
        if _has_unresolved_severity(decision, contradictions, "medium"):
            parts.append(
                _conflict_summary(
                    _caution_drug_name(passages, decision, contradictions, severity="medium")
                )
            )
        if _has_unresolved_severity(decision, contradictions, "low"):
            parts.append(
                _caution_summary(
                    _caution_drug_name(passages, decision, contradictions, severity="qualifier")
                )
            )
        summary = " ".join(parts) if parts else "The available evidence is insufficient."
    detail = _detail(
        accepted=accepted,
        synthesis_failed=synthesis_failed,
        reason=decision.reason if decision else "No sufficiency decision was recorded.",
        missing=missing,
        passage_ids=passage_ids,
        contradiction_count=len(contradictions),
        qualifying=_has_unresolved_severity(decision, contradictions, "low"),
        caution_name=_caution_drug_name(passages, decision, contradictions, severity="qualifier"),
        conflict_drug=_caution_drug_name(passages, decision, contradictions, severity="high"),
        medium_name=_caution_drug_name(passages, decision, contradictions, severity="medium"),
        chart_caution=_chart_caution_line(decision, contradictions),
        evidence_caution=_evidence_caution_line(passages, decision, contradictions),
    )
    card: Dict[str, Any] = {
        "uuid": response.trace_id or str(uuid.uuid4()),
        "summary": summary,
        "indicator": indicator,
        "source": {"label": SOURCE_LABEL},
        "detail": detail,
    }
    extension = {
        "trace_id": response.trace_id,
        "hookInstance": hook_instance,
        "patientId": patient_id,
    }
    card["extension"] = {key: value for key, value in extension.items() if value}
    return [card]


def feedback_documents(
    service_id: str,
    items: Sequence[Any],
    recorded_at: str,
    *,
    user_id: str | None = None,
    org_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Shape one CDS Hooks feedback body into audit documents."""
    documents: List[Dict[str, Any]] = []
    for item in items:
        reason = item.overrideReason
        coded = reason.reason if reason and isinstance(reason.reason, dict) else {}
        suggestions = []
        for suggestion in item.acceptedSuggestions or []:
            if isinstance(suggestion, dict) and suggestion.get("id"):
                suggestions.append(str(suggestion["id"]))
        comment = redact_phi_text(reason.userComment) if reason and reason.userComment else None
        document: Dict[str, Any] = {
            "service_id": service_id,
            "card": item.card,
            "outcome": item.outcome,
            "accepted_suggestions": suggestions,
            "override_code": coded.get("code"),
            "override_comment": comment,
            "outcome_timestamp": item.outcomeTimestamp,
            "recorded_at": recorded_at,
        }
        if user_id is not None:
            document["user_id"] = user_id
        if org_id is not None:
            document["org_id"] = org_id
        documents.append(document)
    return documents


def _clinical_summary(answer: str) -> str:
    """Show the clinical answer. The card detail already carries the boundary."""
    compact = " ".join((answer or "").split())
    marker = "Clinician decision support only:"
    if compact.startswith(marker):
        rest = compact[len(marker):].lstrip()
        period = rest.find(".")
        if period >= 0:
            rest = rest[period + 1 :].lstrip()
        compact = rest
    return _clip(compact or "Evidence passed the MedSwin gate.")


def _clip(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= _SUMMARY_LIMIT:
        return compact
    return compact[: _SUMMARY_LIMIT - 1].rstrip() + "…"


def _chart_caution_line(decision: Any, contradictions: List[Any]) -> str:
    """Return the chart caution sentence stored on a chart conflict."""
    from app.medswin.ledger import PATIENT_CHART_ID

    pairs = list(contradictions or [])
    if decision is not None:
        pairs.extend(list(getattr(decision, "contradictions", []) or []))
    for item in pairs:
        if getattr(item, "resolved", False):
            continue
        if PATIENT_CHART_ID not in {item.chunk_id_a, item.chunk_id_b}:
            continue
        reason = item.reason or ""
        if " | " not in reason or not reason.startswith("Chart caution:"):
            continue
        return reason.split(" | ", 1)[0].strip()
    return ""


def _evidence_caution_line(passages: List[Any], decision: Any, contradictions: List[Any]) -> str:
    """Return the caution sentence from a cited evidence passage."""
    import re

    from app.retrieval.lexical import LexicalRetriever
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    pairs = list(contradictions or [])
    if decision is not None:
        pairs.extend(list(getattr(decision, "contradictions", []) or []))
    cited = set()
    for item in pairs:
        if getattr(item, "resolved", False):
            continue
        cited.add(item.chunk_id_a)
        cited.add(item.chunk_id_b)
    spans: List[str] = []
    for passage in passages or []:
        if getattr(passage, "chunk_id", None) not in cited:
            continue
        text = getattr(passage, "text", "") or ""
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if not sentence.strip():
                continue
            if not (_ungoverned_caution(sentence) or _QUALIFY.search(sentence)):
                continue
            span = LexicalRetriever._caution_span(sentence).strip()
            if span and span not in spans:
                spans.append(span)
    return " ".join(spans)


def _named_summary(drugs: str, full_prefix: str, full_suffix: str, short_prefix: str) -> str:
    """Keep every drug name. Shorten the sentence when the full line exceeds the card limit."""
    full = f"{full_prefix}{drugs}{full_suffix}"
    if len(full) <= _SUMMARY_LIMIT:
        return full
    short = f"{short_prefix}{drugs}."
    if len(short) <= _SUMMARY_LIMIT:
        return short
    names = [name.strip() for name in drugs.split(",") if name.strip()]
    kept: List[str] = []
    for name in names:
        trial = f"{short_prefix}{', '.join(kept + [name])}."
        if len(trial) > _SUMMARY_LIMIT:
            break
        kept.append(name)
    if not kept:
        return _clip(full)
    return f"{short_prefix}{', '.join(kept)}."


def _conflict_summary(drug_name: str) -> str:
    """Name a medium Avoid warning. This sentence is not a low qualifier."""
    if drug_name:
        return _named_summary(
            drug_name,
            "An unresolved caution conflicts with ",
            ".",
            "Conflicts with ",
        )
    return "An unresolved caution conflicts with this recommendation."


def _caution_summary(drug_name: str) -> str:
    """Name the caution drug when the passage states one."""
    if drug_name:
        return _named_summary(
            drug_name,
            "An unresolved caution qualifies ",
            ".",
            "Qualifies ",
        )
    return "An unresolved caution qualifies this recommendation."


def _caution_drug_name(
    passages: List[Any],
    decision: Any,
    contradictions: List[Any],
    severity: str = "low",
) -> str:
    """Return the drugs named by an unresolved caution passage."""
    from app.medswin.ledger import intervention_tokens
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    ids = set()
    pairs = list(contradictions or [])
    if decision is not None:
        pairs.extend(list(getattr(decision, "contradictions", []) or []))
    for item in pairs:
        if item.resolved or not _pair_matches_severity(item, severity):
            continue
        ids.add(item.chunk_id_a)
        ids.add(item.chunk_id_b)
    found: List[str] = []
    for passage in passages or []:
        if getattr(passage, "chunk_id", None) not in ids:
            continue
        text = getattr(passage, "text", "") or ""
        if not (_ungoverned_caution(text) or _QUALIFY.search(text)):
            continue
        for drug in sorted(intervention_tokens(text)):
            if drug not in found:
                found.append(drug)
    if not found:
        from app.medswin.ledger import PATIENT_CHART_ID

        known = {getattr(passage, "chunk_id", None) for passage in passages or []}
        for item in pairs:
            if item.resolved or not _pair_matches_severity(item, severity):
                continue
            cited = {item.chunk_id_a, item.chunk_id_b}
            if cited <= known and PATIENT_CHART_ID not in cited:
                continue
            for drug in sorted(intervention_tokens(getattr(item, "reason", "") or "")):
                if drug not in found:
                    found.append(drug)
    return ", ".join(found)


def _pair_matches_severity(item: Any, severity: str) -> bool:
    """Match one pair. The low label still includes a medium pair."""
    value = getattr(item, "severity", "")
    if severity == "high":
        return value == "high"
    if severity == "medium":
        return value == "medium"
    if severity == "qualifier":
        return value == "low"
    return value != "high"


def _has_unresolved_severity(decision: Any, contradictions: List[Any], severity: str) -> bool:
    """True when an unresolved pair has this severity."""
    pairs = list(contradictions or [])
    if decision is not None:
        pairs.extend(list(getattr(decision, "contradictions", []) or []))
    return any(not item.resolved and getattr(item, "severity", "") == severity for item in pairs)


def _has_unresolved_caution(decision: Any, contradictions: List[Any]) -> bool:
    """True when a non-high pair still blocks the card."""
    pairs = list(contradictions or [])
    if decision is not None:
        pairs.extend(list(getattr(decision, "contradictions", []) or []))
    return any(not item.resolved and item.severity != "high" for item in pairs)


def _detail(
    *,
    accepted: bool,
    reason: str,
    missing: List[str],
    passage_ids: List[str],
    contradiction_count: int,
    synthesis_failed: bool = False,
    qualifying: bool = False,
    caution_name: str = "",
    conflict_drug: str = "",
    medium_name: str = "",
    chart_caution: str = "",
    evidence_caution: str = "",
) -> str:
    missing_text = ", ".join(missing) if missing else "none"
    passage_text = ", ".join(passage_ids) if passage_ids else "none"
    if synthesis_failed:
        outcome = (
            "The evidence gate accepted the passages. "
            "The model did not return a structured clinical answer."
        )
    elif accepted:
        outcome = "The evidence gate accepted this card."
    else:
        outcome = "The evidence gate did not accept this card."
    lead = outcome if synthesis_failed else f"{outcome} {reason}"
    if qualifying and "unresolved caution" not in reason.lower():
        lead = f"{lead} {_caution_summary(caution_name)}"
    if conflict_drug and conflict_drug.lower() not in reason.lower():
        lead = f"{lead} Unresolved evidence conflicts for {conflict_drug} need clinician review."
    if medium_name:
        medium_sentence = _conflict_summary(medium_name)
        if medium_sentence.lower() not in lead.lower():
            lead = f"{lead} {medium_sentence}"
    if chart_caution and chart_caution.lower() not in lead.lower():
        sentence = chart_caution if chart_caution.endswith(".") else f"{chart_caution}."
        lead = f"{lead} {sentence}"
    if evidence_caution and evidence_caution.lower() not in lead.lower():
        sentence = evidence_caution if evidence_caution.endswith(".") else f"{evidence_caution}."
        lead = f"{lead} {sentence}"
    return (
        f"{lead} "
        f"Missing facets: {missing_text}. "
        f"Contradictions: {contradiction_count}. "
        f"Passage ids: {passage_text}. "
        "Clinician decision support only. This card does not establish a diagnosis or a treatment order."
    )
