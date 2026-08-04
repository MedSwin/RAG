"""Scoring: fusion, utility, coverage, calibration, evidence hierarchy."""

from app.scoring.calibrate import CalibrationStore, apply_calibration
from app.scoring.coverage import compute_facet_coverage, score_passage_facets
from app.scoring.fusion import compute_fusion_scores
from app.scoring.hierarchy import evidence_grade_from_metadata
from app.scoring.utility import select_bundle

__all__ = [
    "CalibrationStore",
    "apply_calibration",
    "compute_facet_coverage",
    "compute_fusion_scores",
    "evidence_grade_from_metadata",
    "score_passage_facets",
    "select_bundle",
]
