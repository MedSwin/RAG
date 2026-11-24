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


def test_the_named_eval_portfolio_is_outside_the_unit_gate():
    from pathlib import Path

    names = ("clindet", "medabstain", "adversemed", "basics-cdss", "basics_cdss")
    roots = (Path("app"), Path("tests"), Path("benchmarks"), Path("scripts"), Path(".github"))
    hits = []
    here = Path(__file__).resolve()
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".yml", ".yaml", ".sh"}:
                continue
            if path.resolve() == here:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            hits.extend(f"{path}:{name}" for name in names if name in text)
    assert hits == []
    workflow = Path(".github/workflows/system-eval-contract.yml").read_text(encoding="utf-8")
    assert "pytest -q" in workflow
    assert "verify_live_fullstack" not in workflow
    assert "smoke_realworld" not in workflow
