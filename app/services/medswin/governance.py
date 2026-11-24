"""Governance helpers for MedSwin traces and clinician CDS output."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.schemas.evidence import CandidatePassage, EvidenceGrade
from app.scoring.calibrate import first_present
from app.scoring.hierarchy import evidence_grade_from_metadata as _hierarchy_grade
from app.services.adapters.limiter import rate_limit_snapshot


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")
_MRN_RE = re.compile(
    r"\b(?:MRN|medicare|patient(?:\s+id)?)\s*[:#-]\s*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)
_DOB_RE = re.compile(r"\b(?:DOB|date of birth)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a score to a bounded interval."""
    return max(lower, min(upper, float(value)))


def redact_phi_text(text: str) -> str:
    """Redact common direct identifiers from trace/log text."""
    if not text:
        return text
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _MRN_RE.sub("[REDACTED_PATIENT_ID]", redacted)
    redacted = _DOB_RE.sub("[REDACTED_DOB]", redacted)
    return redacted


_TRACE_IDENTITY_KEYS = {
    "created_at",
    "completed_at",
    "timestamp",
    "trace_id",
    "session_id",
    "chunk_id",
    "doc_id",
    "org_id",
    "user_id",
    "patient_id",
    "hook_instance",
    "generator_profile",
    "generator_model",
}


def lead_with_caution(text: str) -> str:
    """Put the caution where the clinician sees the first lines of a passage."""
    from app.retrieval.lexical import LexicalRetriever
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    raw = text or ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", raw) if part.strip()]

    def _is_caution(sentence: str) -> bool:
        return bool(_ungoverned_caution(sentence) or _QUALIFY.search(sentence))

    def _lead_sentence(sentence: str) -> str:
        span = LexicalRetriever._caution_span(sentence)
        index = sentence.lower().find(span.lower()) if span else 0
        if span and (index < 0 or index > 80):
            return f"{span} {sentence}"
        return sentence

    cautions = [sentence for sentence in sentences if _is_caution(sentence)]
    if len(sentences) > 1 and cautions and not _is_caution(sentences[0]):
        rest = [sentence for sentence in sentences if not _is_caution(sentence)]
        ordered = " ".join([_lead_sentence(sentence) for sentence in cautions] + rest)
    elif len(sentences) > 1 and cautions:
        ordered = " ".join(
            _lead_sentence(sentence) if _is_caution(sentence) else sentence for sentence in sentences
        )
    elif len(sentences) > 1:
        ordered = raw
    else:
        ordered = _lead_sentence(raw)
    return _prepend_distant_drug(_prepend_dose(ordered))


def _prepend_distant_drug(text: str) -> str:
    """Put a named drug in the first lines when a long note would hide it."""
    from app.medswin.ledger import _DRUG_ALIASES, _SHORT_ALIASES, intervention_tokens

    raw = text or ""
    head = raw[:80].lower()
    aliases = {**_DRUG_ALIASES, **_SHORT_ALIASES}
    missing = []
    for drug in sorted(intervention_tokens(raw)):
        if drug in head:
            continue
        if any(re.search(rf"\b{re.escape(alias)}\b", head) for alias, generic in aliases.items() if generic == drug):
            continue
        missing.append(drug)
    if not missing:
        return raw
    return f"{' '.join(missing)} {raw}".strip()


def _prepend_dose(text: str) -> str:
    """Put a chart dose in the first lines when it is not a caution sentence."""
    from app.medswin.ledger import dose_span

    raw = text or ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", raw) if part.strip()]
    doses = [sentence for sentence in sentences if dose_span(sentence).lower() == sentence.lower()]
    if len(sentences) > 1 and doses and sentences[0] not in doses:
        rest = [sentence for sentence in sentences if sentence not in doses]
        raw = " ".join(doses + rest)
        sentences = doses + rest
    extras = []
    for sentence in sentences:
        span = dose_span(sentence)
        if not span or span.lower() == sentence.lower():
            continue
        if raw.lower().find(span.lower()) > 80 and span not in extras:
            extras.append(span)
    if extras:
        return f"{' '.join(extras)} {raw}".strip()
    return raw


def _lead_passages(payload: Dict[str, Any]) -> None:
    bundle = payload.get("evidence_bundle") or {}
    for passage in bundle.get("passages") or []:
        if isinstance(passage, dict) and isinstance(passage.get("text"), str):
            passage["text"] = lead_with_caution(passage["text"])


def redact_public_chat(response: Any) -> Any:
    """Redact a chat response after the gate has finished."""
    from app.schemas.traces import ChatResponse

    payload = redact_stored_trace(response.model_dump(mode="python"))
    _lead_passages(payload)
    return ChatResponse.model_validate(payload)


def redact_stored_trace(payload: Any) -> Any:
    """Redact free text in a stored trace. Keep ids and timestamps."""
    return _redact_stored(payload, None)


def _redact_stored(payload: Any, key: str | None) -> Any:
    if key in _TRACE_IDENTITY_KEYS:
        return payload
    if isinstance(payload, str):
        return redact_phi_text(payload)
    if isinstance(payload, list):
        return [_redact_stored(item, None) for item in payload]
    if isinstance(payload, dict):
        return {item_key: _redact_stored(value, item_key) for item_key, value in payload.items()}
    return payload


def redact_phi_payload(payload: Any) -> Any:
    """Recursively redact strings inside JSON-like trace payloads."""
    if isinstance(payload, str):
        return redact_phi_text(payload)
    if isinstance(payload, list):
        return [redact_phi_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: redact_phi_payload(value) for key, value in payload.items()}
    return payload


def evidence_grade_from_metadata(candidate: CandidatePassage) -> EvidenceGrade:
    """Derive an evidence grade from explicit metadata and source type."""
    return _hierarchy_grade(candidate)


def build_citation(candidate: CandidatePassage, facets: Iterable[str] = ()) -> Dict[str, Any]:
    """Build a stable citation object with provenance and policy metadata."""
    metadata = candidate.metadata or {}
    return {
        "chunk_id": candidate.chunk_id,
        "doc_id": candidate.doc_id,
        "source_type": candidate.source_type.value,
        "section": candidate.section,
        "offset_start": candidate.offset_start,
        "offset_end": candidate.offset_end,
        "guideline_version": metadata.get("guideline_version") or metadata.get("version"),
        "effective_date": metadata.get("effective_date"),
        "timestamp": metadata.get("timestamp"),
        "facets": list(facets),
        "calibrated_relevance": first_present(candidate.calibrated_score, candidate.rerank_score),
        "fusion_score": candidate.fusion_score,
        "evidence_grade": evidence_grade_from_metadata(candidate).model_dump(),
    }


def redacted_trace_summary(trace: Dict[str, Any], include_policy_details: bool = False) -> Dict[str, Any]:
    """Return a PHI-safe trace summary for API responses."""
    evidence_bundle = trace.get("evidence_bundle") or {}
    summary = {
        "trace_id": trace.get("trace_id"),
        "session_id": trace.get("session_id"),
        "query": redact_phi_text(trace.get("query", "")),
        "created_at": trace.get("created_at"),
        "completed_at": trace.get("completed_at"),
        "messages_count": len(trace.get("messages", [])),
        "tool_calls_count": len(trace.get("tool_calls", [])),
        "sufficiency_checks_count": len(trace.get("sufficiency_checks", [])),
        "evidence_passages_count": len(evidence_bundle.get("passages", [])),
    }
    # Motivation vs Logic: reranker and embedding backoffs are operational
    # state, not PHI. Exposing the snapshot lets the benchmark audit quota
    # pressure without scraping logs or guessing from wall-clock time.
    try:
        summary["rate_limit_stats"] = rate_limit_snapshot()
    except Exception:
        summary["rate_limit_stats"] = {}
    if include_policy_details:
        summary["policy_decisions"] = redact_phi_payload(trace.get("policy_decisions", []))
        summary["facet_coverage"] = redact_phi_payload(trace.get("facet_coverage", []))
        summary["contradictions"] = redact_phi_payload(trace.get("contradictions", []))
        summary["facet_matrix"] = redact_phi_payload(trace.get("facet_matrix"))
        summary["contradiction_ledger"] = redact_phi_payload(trace.get("contradiction_ledger"))
        summary["sufficiency_decision"] = redact_phi_payload(trace.get("sufficiency_decision"))
        summary["answer_provenance"] = redact_phi_payload(trace.get("answer_provenance"))
        summary["retrieval_traces"] = redact_phi_payload(trace.get("retrieval_traces", []))
        summary["rerank_traces"] = redact_phi_payload(trace.get("rerank_traces", []))
        summary["evidence_ledger"] = redact_phi_payload(trace.get("evidence_ledger", []))
        summary["tool_calls"] = redact_phi_payload(trace.get("tool_calls", []))
        summary["final_answer"] = redact_phi_text(trace.get("final_answer") or "")
        summary["citations"] = redact_phi_payload(trace.get("citations", []))
        summary["degraded_mode"] = trace.get("degraded_mode") or {}
    return summary


def ensure_cds_language(answer: str) -> str:
    """Make the clinical boundary explicit for final responses."""
    boundary = (
        "Clinician decision support only: this response supports clinical review and "
        "does not establish a final diagnosis."
    )
    if boundary.lower() in answer.lower():
        return answer
    return f"{boundary}\n\n{answer}"
