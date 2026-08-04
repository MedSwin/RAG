"""Mongo retrieval filter construction."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.config import settings
from app.schemas.enums import SourceType


def retrieval_filter(
    org_id: str,
    source_type_filter: Optional[SourceType] = None,
    patient_id: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build Mongo retrieval filters from policy constraints."""
    constraints = constraints or {}
    filter_dict: Dict[str, Any] = {"org_id": org_id}

    source_policy = constraints.get("source_policy")
    if source_type_filter:
        filter_dict["source_type"] = source_type_filter.value
    elif source_policy and source_policy != "ANY":
        source_policy_map = {
            "CPG_ONLY": SourceType.CPG.value,
            "EMR_ONLY": SourceType.EMR.value,
            "LIT_ONLY": SourceType.LIT.value,
            "SAFETY_ONLY": SourceType.SAFETY.value,
            SourceType.CPG.value: SourceType.CPG.value,
            SourceType.EMR.value: SourceType.EMR.value,
            SourceType.LIT.value: SourceType.LIT.value,
            SourceType.SAFETY.value: SourceType.SAFETY.value,
        }
        mapped = source_policy_map.get(str(source_policy).upper())
        if mapped:
            filter_dict["source_type"] = mapped

    if patient_id and (
        source_type_filter == SourceType.EMR
        or str(source_policy).upper() == "EMR_ONLY"
        or constraints.get("patient_scope_only") is True
    ):
        filter_dict["patient_id"] = patient_id
    elif patient_id:
        filter_dict["$and"] = filter_dict.get("$and", []) + [
            {"$or": [
                {"source_type": {"$ne": SourceType.EMR.value}},
                {"patient_id": patient_id},
            ]}
        ]

    if settings.CLOUD_MODE:
        filter_dict["embedding_space"] = settings.active_embedding_space()
        filter_dict["embedding_model"] = settings.CLOUD_EMBEDDING
        filter_dict["embedding_dim"] = settings.active_embedding_dimension()

    min_grade = constraints.get("min_evidence_grade")
    if min_grade is not None:
        try:
            min_grade_value = float(min_grade)
            filter_dict["$or"] = [
                {"evidence_grade.score": {"$gte": min_grade_value}},
                {"metadata.evidence_grade.score": {"$gte": min_grade_value}},
                {"source_reliability": {"$gte": min_grade_value}},
            ]
        except (TypeError, ValueError):
            pass

    timeframe = constraints.get("timeframe")
    if isinstance(timeframe, dict):
        date_filter = {}
        if timeframe.get("start"):
            date_filter["$gte"] = timeframe["start"]
        if timeframe.get("end"):
            date_filter["$lte"] = timeframe["end"]
        if date_filter:
            filter_dict["$and"] = filter_dict.get("$and", []) + [
                {"$or": [{"timestamp": date_filter}, {"metadata.effective_date": date_filter}]}
            ]
    elif isinstance(timeframe, str) and len(timeframe) == 4 and timeframe.isdigit():
        filter_dict["$and"] = filter_dict.get("$and", []) + [
            {"$or": [
                {"timestamp": {"$gte": f"{timeframe}-01-01", "$lte": f"{timeframe}-12-31"}},
                {"metadata.effective_date": {"$gte": f"{timeframe}-01-01", "$lte": f"{timeframe}-12-31"}},
            ]}
        ]

    specialties = constraints.get("specialties") or []
    if specialties:
        filter_dict["$and"] = filter_dict.get("$and", []) + [
            {"$or": [
                {"tags": {"$in": specialties}},
                {"metadata.specialty": {"$in": specialties}},
                {"metadata.specialties": {"$in": specialties}},
            ]}
        ]

    if constraints.get("safety_search"):
        filter_dict["$and"] = filter_dict.get("$and", []) + [
            {"$or": [
                {"source_type": SourceType.SAFETY.value},
                {"section": {"$regex": "contraindicat|warning|adverse|interaction", "$options": "i"}},
                {"text": {"$regex": "contraindicat|adverse|interaction|avoid", "$options": "i"}},
            ]}
        ]

    return filter_dict
