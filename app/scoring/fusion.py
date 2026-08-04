"""Policy-aware fusion on the log-odds scale."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

from app.core.config import settings
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from app.scoring.calibrate import clamp
from app.scoring.hierarchy import evidence_grade_from_metadata


def _clip_logit(value: float) -> float:
    return max(-settings.FUSION_LOGIT_CLIP, min(settings.FUSION_LOGIT_CLIP, value))


def _recency(candidate: CandidatePassage) -> float:
    timestamp = candidate.metadata.get("timestamp") or candidate.metadata.get("effective_date")
    if not timestamp:
        return 0.5
    try:
        if isinstance(timestamp, str):
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif isinstance(timestamp, datetime):
            parsed = timestamp
        else:
            return 0.5
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - parsed).days, 0)
        return clamp(math.exp(-age_days / max(settings.RECENCY_DECAY_DAYS, 1.0)))
    except Exception:
        return 0.5


def _section(candidate: CandidatePassage) -> float:
    section = (candidate.section or "").lower()
    if any(term in section for term in ("recommendation", "guideline", "contraindication", "warning")):
        return settings.SECTION_RECOMMENDATION_SCORE
    if any(term in section for term in ("background", "introduction")):
        return settings.SECTION_BACKGROUND_SCORE
    return settings.SECTION_DEFAULT_SCORE


def _source(candidate: CandidatePassage) -> float:
    if candidate.source_type == SourceType.CPG:
        return settings.SOURCE_CPG_SCORE
    if candidate.source_type == SourceType.EMR:
        return settings.SOURCE_EMR_SCORE
    if candidate.source_type == SourceType.SAFETY:
        return settings.EBM_SAFETY_WEIGHT
    return settings.SOURCE_LIT_SCORE


def _safety(candidate: CandidatePassage) -> float:
    text = candidate.text.lower()
    terms = ["contraindicat", "allergy", "adverse", "interaction", "avoid", "renal", "pregnan", "dose"]
    return clamp(sum(1 for term in terms if term in text) * 0.16)


def _noise(candidate: CandidatePassage) -> float:
    metadata = candidate.metadata or {}
    score = 0.0
    if metadata.get("obsolete") or metadata.get("superseded"):
        score += 0.45
    if metadata.get("population_mismatch"):
        score += 0.35
    if metadata.get("weak_provenance"):
        score += 0.25
    if not candidate.doc_id or not candidate.chunk_id:
        score += 0.15
    return clamp(score)


def _contradiction(candidate: CandidatePassage) -> float:
    text = candidate.text.lower()
    terms = ["conflict", "not recommended", "avoid", "insufficient", "uncertain", "contraindicat"]
    return clamp(sum(1 for term in terms if term in text) * 0.16)


def compute_fusion_scores(candidates: List[CandidatePassage]) -> List[CandidatePassage]:
    """Compute fused clinical scores for candidates."""
    for candidate in candidates:
        p_hat = clamp(candidate.calibrated_score or candidate.rerank_score or 0.50, 1e-6, 1.0 - 1e-6)
        rerank_log_odds = _clip_logit(math.log(p_hat / (1.0 - p_hat)))
        dense = clamp(candidate.dense_score or 0.0)
        lexical = clamp(candidate.lexical_score or 0.0)
        recency_score = _recency(candidate)
        section_score = _section(candidate)
        source_score = _source(candidate)
        evidence_grade = evidence_grade_from_metadata(candidate)
        ebm_score = clamp(evidence_grade.score)
        safety_score = _safety(candidate)
        noise_score = _noise(candidate)

        raw = (
            settings.W_RERANK * rerank_log_odds
            + settings.W_DENSE * dense
            + settings.W_LEX * lexical
            + settings.W_RECENCY * recency_score
            + settings.W_SECTION * section_score
            + settings.W_SOURCE * source_score
            + settings.W_EBM * ebm_score
            + settings.SAFETY_REWARD_WEIGHT * safety_score
            - settings.W_NOISE * noise_score
        )
        candidate.recency_score = recency_score
        candidate.section_score = section_score
        candidate.source_score = source_score
        candidate.evidence_grade_score = ebm_score
        candidate.safety_score = safety_score
        candidate.noise_score = noise_score
        candidate.contradiction_score = _contradiction(candidate)
        candidate.fusion_score = clamp(1.0 / (1.0 + math.exp(-_clip_logit(raw))))

    candidates.sort(key=lambda item: item.fusion_score or 0.0, reverse=True)
    return candidates
