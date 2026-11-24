from app.agents.base import parse_claims
from app.medswin.ledger import build_retrieval_ledger
from app.schemas.enums import SourceType
from app.schemas.evidence import CandidatePassage
from app.scoring.fusion import compute_fusion_scores
from app.services.medswin.governance import build_citation


def _passage(**scores) -> CandidatePassage:
    return CandidatePassage(
        chunk_id=scores.pop("chunk_id", "c1"),
        doc_id="d1",
        source_type=SourceType.CPG,
        text="Guideline recommendation supports treatment.",
        section="Recommendations",
        **scores,
    )


def test_fusion_keeps_a_zero_calibrated_score_below_a_modest_score():
    zero = _passage(chunk_id="zero", calibrated_score=0.0, rerank_score=0.95, dense_score=0.2)
    modest = _passage(chunk_id="modest", calibrated_score=0.40, rerank_score=0.40, dense_score=0.2)
    scored = compute_fusion_scores([zero, modest])
    assert scored[0].chunk_id == "modest"


def test_ledger_and_citation_keep_a_zero_calibrated_score():
    passage = _passage(calibrated_score=0.0, rerank_score=0.9, fusion_score=0.9)
    entry = build_retrieval_ledger([passage], [])[0]
    assert entry.calibrated_relevance == 0.0
    assert build_citation(passage)["calibrated_relevance"] == 0.0


def test_a_later_zero_calibration_replaces_the_earlier_score():
    from app.medswin.orchestrator import merge_candidate_scores

    existing = _passage(calibrated_score=0.8, rerank_score=0.8, dense_score=0.4)
    incoming = _passage(calibrated_score=0.0, rerank_score=0.0)
    merge_candidate_scores(existing, incoming)
    assert existing.calibrated_score == 0.0
    assert existing.rerank_score == 0.0
    assert existing.dense_score == 0.4


def test_a_zero_claim_confidence_stays_zero():
    passage = _passage(calibrated_score=0.0, rerank_score=0.9)
    batch = parse_claims(
        "safety",
        {"claims": [{"facet": "safety", "claim": "Avoid this dose.", "chunk_id": "c1", "polarity": "safety", "confidence": 0}]},
        [passage],
    )
    assert batch.claims[0].confidence == 0.0
    assert batch.claims[0].calibrated_relevance == 0.0
