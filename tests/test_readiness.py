from app.medswin.readiness import cloud_health_status


def test_missing_index_stays_healthy():
    assert cloud_health_status(index_exists=False, provenance_valid=False) == "healthy"


def test_matching_index_stays_healthy():
    assert cloud_health_status(index_exists=True, provenance_valid=True) == "healthy"


def test_wrong_embedding_space_is_degraded():
    assert cloud_health_status(index_exists=True, provenance_valid=False) == "degraded"
