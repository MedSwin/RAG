from app.core.config import settings


def test_default_fusion_weights_include_evidence_and_safety():
    assert settings.validate_fusion_weights() is True
    assert settings.fusion_positive_weight_sum() > (
        settings.W_RERANK + settings.W_DENSE + settings.W_LEX
        + settings.W_RECENCY + settings.W_SECTION + settings.W_SOURCE
    )


def test_a_large_evidence_weight_fails_validation(monkeypatch):
    monkeypatch.setattr(settings, "W_EBM", 5.0)
    assert settings.validate_fusion_weights() is False


def test_a_negative_noise_weight_fails_validation(monkeypatch):
    monkeypatch.setattr(settings, "W_NOISE", -0.2)
    assert settings.validate_fusion_weights() is False


def test_an_invalid_fusion_weight_stops_startup(monkeypatch):
    monkeypatch.setattr(settings, "W_EBM", 5.0)
    try:
        settings.enforce_fusion_weights()
    except ValueError as exc:
        assert "Fusion weights" in str(exc)
    else:
        raise AssertionError("An invalid fusion weight was allowed to start")
