"""Governance helpers for MedSwin traces and clinician CDS output."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.schemas.evidence import CandidatePassage, EvidenceGrade
from app.scoring.hierarchy import evidence_grade_from_metadata as _hierarchy_grade
from app.services.adapters.limiter import rate_limit_snapshot


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")
_MRN_RE = re.compile(r"\b(?:MRN|patient(?:\s+id)?|medicare)\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.IGNORECASE)
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
        "calibrated_relevance": candidate.calibrated_score or candidate.rerank_score,
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
