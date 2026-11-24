from app.retrieval.filters import retrieval_filter
from app.schemas.enums import SourceType


def test_a_query_without_a_patient_excludes_charts():
    query_filter = retrieval_filter("org-1", None, None, {})
    assert {"source_type": {"$ne": SourceType.EMR.value}} in query_filter["$and"]


def test_a_chart_search_without_a_patient_matches_nothing():
    query_filter = retrieval_filter("org-1", SourceType.EMR, None, {})
    assert {"patient_id": {"$in": []}} in query_filter["$and"]


def test_a_named_patient_keeps_that_chart_and_shared_evidence():
    query_filter = retrieval_filter("org-1", None, "patient-1", {})
    clause = query_filter["$and"][0]["$or"]
    assert {"patient_id": "patient-1"} in clause
    assert {"source_type": {"$ne": SourceType.EMR.value}} in clause
