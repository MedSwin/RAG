from app.medswin.readiness import cloud_health_status


def test_missing_index_stays_healthy():
    assert cloud_health_status(index_exists=False, provenance_valid=False) == "healthy"


def test_matching_index_stays_healthy():
    assert cloud_health_status(index_exists=True, provenance_valid=True) == "healthy"


def test_wrong_embedding_space_is_degraded():
    assert cloud_health_status(index_exists=True, provenance_valid=False) == "degraded"


def test_enable_otel_does_not_steer_the_clinical_path():
    from pathlib import Path

    from app.core.config import settings

    assert settings.ENABLE_OTEL is False
    hits = sorted(
        str(path.relative_to(Path("app")))
        for path in Path("app").rglob("*.py")
        if "ENABLE_OTEL" in path.read_text(encoding="utf-8")
    )
    assert hits == ["core/config.py"]
    assert "opentelemetry" not in Path("requirements.txt").read_text(encoding="utf-8").lower()
