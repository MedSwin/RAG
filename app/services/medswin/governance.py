"""Governance helpers for MedSwin traces and clinician CDS output."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.models.medswin import CandidatePassage, EvidenceGrade, SourceType
from app.services.adapters.rate_limit import rate_limit_snapshot


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
    metadata = candidate.metadata or {}
    raw_grade = metadata.get("evidence_grade")
    if isinstance(raw_grade, dict):
        return EvidenceGrade(**raw_grade)
    if isinstance(candidate.evidence_grade_score, (int, float)):
        return EvidenceGrade(
            label=str(raw_grade or "metadata_score"),
            score=clamp(candidate.evidence_grade_score),
            source_reliability=clamp(metadata.get("source_reliability", 0.5)),
        )

    label = str(raw_grade or metadata.get("study_design") or candidate.source_type.value).lower()
    score_map = {
        "cpg": 0.95,
        "guideline": 0.95,
        "systematic_review": 0.90,
        "sr": 0.90,
        "rct": 0.86,
        "trial": 0.80,
        "observational": 0.62,
        "obs": 0.62,
        "case": 0.38,
        "emr": 0.70,
        "safety": 0.88,
    }
    fallback = {
        SourceType.CPG: 0.95,
        SourceType.EMR: 0.70,
        SourceType.LIT: 0.75,
    }.get(candidate.source_type, 0.50)
    score = next((value for key, value in score_map.items() if key in label), fallback)
    return EvidenceGrade(
        label=label or "ungraded",
        score=clamp(score),
        source_reliability=clamp(metadata.get("source_reliability", score)),
        rationale=metadata.get("evidence_rationale"),
    )


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
