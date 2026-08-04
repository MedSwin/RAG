"""Evidence hierarchy / EBM scoring."""

from __future__ import annotations

from typing import Dict

from app.core.config import settings
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage, EvidenceGrade
from app.scoring.calibrate import clamp


def evidence_vector(candidate: CandidatePassage) -> Dict[str, float]:
    """Build evidence-grade vector components for a passage."""
    metadata = candidate.metadata or {}
    label = str(
        metadata.get("evidence_grade")
        or metadata.get("study_design")
        or candidate.source_type.value
    ).lower()
    vector = {
        "cpg": 0.0,
        "sr": 0.0,
        "rct": 0.0,
        "obs": 0.0,
        "case": 0.0,
        "emr": 0.0,
        "safety": 0.0,
    }
    if candidate.source_type == SourceType.CPG or "guideline" in label or "cpg" in label:
        vector["cpg"] = 1.0
    elif "systematic" in label or label in {"sr", "systematic_review"}:
        vector["sr"] = 1.0
    elif "rct" in label or "trial" in label:
        vector["rct"] = 1.0
    elif "observ" in label or label == "obs":
        vector["obs"] = 1.0
    elif "case" in label:
        vector["case"] = 1.0
    elif candidate.source_type == SourceType.EMR or "emr" in label:
        vector["emr"] = 1.0
    elif candidate.source_type == SourceType.SAFETY or "safety" in label:
        vector["safety"] = 1.0
    elif candidate.source_type == SourceType.LIT:
        vector["obs"] = 0.5
        vector["rct"] = 0.3
    return vector


def evidence_grade_from_metadata(candidate: CandidatePassage) -> EvidenceGrade:
    """Derive an evidence grade from explicit metadata and source type."""
    metadata = candidate.metadata or {}
    raw_grade = metadata.get("evidence_grade")
    if isinstance(raw_grade, dict):
        grade = EvidenceGrade(**raw_grade)
        if not grade.vector:
            grade.vector = evidence_vector(candidate)
        return grade
    if isinstance(candidate.evidence_grade_score, (int, float)):
        return EvidenceGrade(
            label=str(raw_grade or "metadata_score"),
            score=clamp(candidate.evidence_grade_score),
            source_reliability=clamp(metadata.get("source_reliability", 0.5)),
            vector=evidence_vector(candidate),
        )

    vector = evidence_vector(candidate)
    weights = {
        "cpg": settings.EBM_CPG_WEIGHT,
        "sr": settings.EBM_SR_WEIGHT,
        "rct": settings.EBM_RCT_WEIGHT,
        "obs": settings.EBM_OBS_WEIGHT,
        "case": settings.EBM_CASE_WEIGHT,
        "emr": settings.EBM_EMR_WEIGHT,
        "safety": settings.EBM_SAFETY_WEIGHT,
    }
    score = sum(weights[key] * value for key, value in vector.items())
    if score <= 0.0:
        fallback = {
            SourceType.CPG: settings.EBM_CPG_WEIGHT,
            SourceType.EMR: settings.EBM_EMR_WEIGHT,
            SourceType.LIT: settings.SOURCE_LIT_SCORE,
            SourceType.SAFETY: settings.EBM_SAFETY_WEIGHT,
        }.get(candidate.source_type, 0.50)
        score = fallback
    label = str(raw_grade or metadata.get("study_design") or candidate.source_type.value).lower()
    return EvidenceGrade(
        label=label or "ungraded",
        score=clamp(score),
        source_reliability=clamp(metadata.get("source_reliability", score)),
        rationale=metadata.get("evidence_rationale"),
        vector=vector,
    )
