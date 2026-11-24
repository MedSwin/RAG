from app.core.config import settings
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from app.agents.base import prioritize_passages
from app.scoring.fusion import _contradiction, _safety


def test_a_dose_sentence_does_not_raise_the_safety_rank():
    dose = CandidatePassage(
        chunk_id="dose",
        doc_id="dose",
        source_type=SourceType.CPG,
        text="The metformin dose is 500 mg each morning.",
    )
    caution = CandidatePassage(
        chunk_id="caution",
        doc_id="caution",
        source_type=SourceType.SAFETY,
        text="Avoid metformin in pregnancy when renal function is low.",
    )
    assert _safety(dose) == 0.0
    assert _safety(caution) > 0.0


def test_a_direct_caution_raises_the_safety_rank():
    direct = CandidatePassage(
        chunk_id="direct",
        doc_id="direct",
        source_type=SourceType.SAFETY,
        text="Do not use metformin when the eGFR is low.",
    )
    negated = CandidatePassage(
        chunk_id="negated",
        doc_id="negated",
        source_type=SourceType.CPG,
        text="The guideline reports no contraindications for selected adults.",
    )
    delay = CandidatePassage(
        chunk_id="delay",
        doc_id="delay",
        source_type=SourceType.CPG,
        text="Do not delay antibiotics when infection is confirmed.",
    )
    renal = CandidatePassage(
        chunk_id="renal",
        doc_id="renal",
        source_type=SourceType.CPG,
        text="Metformin needs a renal review.",
        calibrated_score=0.99,
    )
    filler = CandidatePassage(
        chunk_id="filler",
        doc_id="filler",
        source_type=SourceType.CPG,
        text="A general management statement.",
        calibrated_score=0.99,
    )
    assert _safety(direct) >= 0.32
    assert _contradiction(direct) >= 0.32
    assert _safety(negated) == 0.0
    assert _contradiction(negated) == 0.0
    assert _safety(delay) == 0.0
    assert _contradiction(delay) == 0.0
    assert _safety(renal) < 0.32
    direct.safety_score = _safety(direct)
    assert prioritize_passages([filler, direct])[0].chunk_id == "direct"


def test_a_hedge_does_not_pin_a_contradiction():
    hedge = CandidatePassage(
        chunk_id="hedge",
        doc_id="hedge",
        source_type=SourceType.LIT,
        text="The uncertainty is insufficient. Avoidance of delay is still recommended.",
    )
    interest = CandidatePassage(
        chunk_id="interest",
        doc_id="interest",
        source_type=SourceType.LIT,
        text="The conflict of interest statement says the evidence is uncertain and insufficient.",
    )
    caution = CandidatePassage(
        chunk_id="caution",
        doc_id="caution",
        source_type=SourceType.SAFETY,
        text="Avoid metformin because it is not recommended when the eGFR is low.",
    )
    assert _contradiction(hedge) == 0.0
    assert _contradiction(interest) < 0.32
    assert _contradiction(caution) >= 0.32


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
