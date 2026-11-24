from app.api.v1.endpoints.storage import _stale_refresh_filter
from app.retrieval.filters import retrieval_filter
from app.retrieval.lexical import chart_scope_sql
from app.schemas.enums import SourceType


def test_a_query_without_a_patient_excludes_charts():
    query_filter = retrieval_filter("org-1", None, None, {})
    assert {"source_type": {"$ne": SourceType.EMR.value}} in query_filter["$and"]
    assert {"patient_id": {"$in": [None, ""]}} in query_filter["$and"][1]["$or"]


def test_a_chart_search_without_a_patient_matches_nothing():
    query_filter = retrieval_filter("org-1", SourceType.EMR, None, {})
    assert {"patient_id": {"$in": []}} in query_filter["$and"]


def test_a_clinician_refresh_does_not_select_charts():
    clinician = _stale_refresh_filter("org-1", include_charts=False)
    operator = _stale_refresh_filter("org-1", include_charts=True)
    assert clinician["source_type"] == {"$ne": "EMR"}
    assert "source_type" not in operator


def test_a_named_patient_keeps_that_chart_and_shared_evidence():
    query_filter = retrieval_filter("org-1", None, "patient-1", {})
    clause = query_filter["$and"][0]["$or"]
    assert {"patient_id": "patient-1"} in clause
    shared = next(item for item in clause if "$and" in item)
    assert {"source_type": {"$ne": SourceType.EMR.value}} in shared["$and"]
    assert {"patient_id": "other-patient"} not in clause


def test_the_patient_lookup_reads_three_chart_rows():
    import asyncio

    from app.repositories.chunks import ChunkRepository

    seen = {}

    class _Cursor:
        def __init__(self, query):
            self.query = query
            self.bounded = None

        def sort(self, spec):
            seen["sort"] = spec
            return self

        def limit(self, count):
            self.bounded = count
            return self

        async def to_list(self, length=None):
            seen["query"] = self.query
            seen["limit"] = self.bounded
            seen["length"] = length
            return []

    class _Store:
        def find(self, query):
            return _Cursor(query)

    repo = ChunkRepository()
    repo._db = {"chunks": _Store()}
    asyncio.run(repo.get_by_patient_id("pt-1", "org-a", source_type="EMR", limit=3))
    assert seen["query"] == {"patient_id": "pt-1", "org_id": "org-a", "source_type": "EMR"}
    assert seen["sort"] == [("metadata.created_timestamp", -1)]
    assert seen["limit"] == 3
    assert seen["length"] == 3


def test_a_chart_row_is_visible_only_for_its_patient():
    named_sql, named_params = chart_scope_sql("patient-1")
    open_sql, open_params = chart_scope_sql(None)
    assert "patient_id = ?" in named_sql
    assert named_params == ["patient-1"]
    assert "source_type != 'EMR'" in named_sql
    assert "patient_id = ?" not in open_sql
    assert open_params == []
    assert "source_type != 'EMR'" in open_sql
