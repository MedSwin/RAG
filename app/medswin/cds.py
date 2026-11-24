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


def cards_from_chat(response: ChatResponse) -> List[Dict[str, Any]]:
    """Map one chat decision to zero or one CDS card."""
    decision = response.policy_decision
    passages = list(response.evidence_bundle.passages or [])
    passage_ids = [passage.chunk_id for passage in passages if passage.chunk_id]
    missing = list(decision.missing_facets) if decision else []
    contradictions = list(response.contradictions or [])
    unresolved = bool(decision and decision.unresolved_critical_conflicts)
    abstained = bool(response.answer_artifact and response.answer_artifact.abstained)
    accepted = bool(
        decision
        and decision.passed
        and decision.action == PolicyAction.ACCEPT
        and passages
        and not abstained
        and not unresolved
    )
    if accepted:
        indicator = "info"
        summary = _clinical_summary(response.answer)
    elif unresolved:
        indicator = "critical"
        summary = "Unresolved evidence conflicts need clinician review."
    else:
        indicator = "warning"
        summary = "The available evidence is insufficient."
    detail = _detail(
        accepted=accepted,
        reason=decision.reason if decision else "No sufficiency decision was recorded.",
        missing=missing,
        passage_ids=passage_ids,
        contradiction_count=len(contradictions),
    )
    return [
        {
            "uuid": str(uuid.uuid4()),
            "summary": summary,
            "indicator": indicator,
            "source": {"label": SOURCE_LABEL},
            "detail": detail,
        }
    ]


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


def _detail(
    *,
    accepted: bool,
    reason: str,
    missing: List[str],
    passage_ids: List[str],
    contradiction_count: int,
) -> str:
    missing_text = ", ".join(missing) if missing else "none"
    passage_text = ", ".join(passage_ids) if passage_ids else "none"
    outcome = "The evidence gate accepted this card." if accepted else "The evidence gate did not accept this card."
    return (
        f"{outcome} {reason} "
        f"Missing facets: {missing_text}. "
        f"Contradictions: {contradiction_count}. "
        f"Passage ids: {passage_text}. "
        "Clinician decision support only. This card does not establish a diagnosis or a treatment order."
    )
