import numpy as np

from app.models.medswin import CandidatePassage, ClinicalFacet, SourceType
from app.services.medswin.retrieval import RetrievalPipeline


def _candidate(chunk_id, text, source_type=SourceType.CPG, rerank=0.70, dense=0.50):
    return CandidatePassage(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type=source_type,
        text=text,
        section="Recommendations",
        rerank_score=rerank,
        calibrated_score=rerank,
        dense_score=dense,
        lexical_score=0.20,
        metadata={"evidence_grade": {"label": "guideline", "score": 0.95, "source_reliability": 0.95}},
    )


def test_fusion_scores_are_bounded_and_use_calibrated_reranker_signal():
    pipeline = RetrievalPipeline()
    candidates = [
        _candidate("high", "Guideline recommendation with safety review.", rerank=0.95),
        _candidate("low", "Background information.", rerank=0.20),
    ]

    scored = pipeline.compute_fusion_scores(candidates)

    assert scored[0].chunk_id == "high"
    assert all(0.0 <= item.fusion_score <= 1.0 for item in scored)
    assert scored[0].metadata["evidence_grade"]["score"] == 0.95


def test_budgeted_selection_protects_safety_and_limits_redundancy():
    pipeline = RetrievalPipeline()
    facets = [
        ClinicalFacet(name="guideline_concordance", threshold=0.50, keywords=["guideline"]),
        ClinicalFacet(name="safety_contraindications", threshold=0.50, keywords=["contraindication", "avoid"]),
    ]
    candidates = [
        _candidate("guideline", "Guideline recommendation supports first line therapy.", rerank=0.88),
        _candidate("duplicate", "Guideline recommendation supports first line therapy.", rerank=0.86),
        _candidate("safety", "Contraindication: avoid therapy in severe allergy or interaction.", rerank=0.65),
    ]
    scored = pipeline.compute_fusion_scores(candidates)
    for candidate in scored:
        candidate.facet_scores = {
            facet.name: (0.9 if any(keyword in candidate.text.lower() for keyword in facet.keywords) else 0.0)
            for facet in facets
        }

    selected = pipeline.select_with_mmr(scored, np.array([1.0]), max_chunks=2, token_budget=200, facets=facets)
    selected_ids = {item.chunk_id for item in selected}

    assert "safety" in selected_ids
    assert len(selected_ids) == 2


def test_budgeted_selection_preserves_required_literature_source():
    pipeline = RetrievalPipeline()
    facets = [
        ClinicalFacet(
            name="treatment recommendation evidence",
            required=True,
            source_policy="LIT",
            keywords=["treatment", "therapy"],
        ),
        ClinicalFacet(
            name="patient applicability",
            required=True,
            source_policy="EMR",
            keywords=["patient", "history"],
        ),
    ]
    candidates = [
        _candidate("emr", "Patient history and medication context.", source_type=SourceType.EMR, rerank=0.92),
        _candidate("lit", "Treatment therapy evidence from literature.", source_type=SourceType.LIT, rerank=0.64),
        _candidate("background", "Generic background.", source_type=SourceType.CPG, rerank=0.90),
    ]
    scored = pipeline.compute_fusion_scores(candidates)
    for candidate in scored:
        candidate.facet_scores = {
            facet.name: (0.9 if candidate.source_type.value == facet.source_policy else 0.0)
            for facet in facets
        }

    selected = pipeline.select_with_mmr(scored, np.array([1.0]), max_chunks=2, token_budget=200, facets=facets)
    selected_sources = {item.source_type for item in selected}

    assert SourceType.LIT in selected_sources
    assert SourceType.EMR in selected_sources
