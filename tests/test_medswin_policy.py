from app.models.medswin import CandidatePassage, ClinicalFacet, SourceType
from app.services.medswin.policy import EvidenceSufficiencyPolicy


def _passage(chunk_id, text, source_type=SourceType.CPG, score=0.92):
    return CandidatePassage(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type=source_type,
        text=text,
        section="Recommendations",
        rerank_score=score,
        calibrated_score=score,
        fusion_score=score,
        evidence_grade_score=0.95,
    )


def test_facet_sufficiency_passes_with_required_coverage():
    policy = EvidenceSufficiencyPolicy()
    facets = [
        ClinicalFacet(name="guideline_concordance", threshold=0.55, keywords=["guideline", "recommendation"]),
        ClinicalFacet(name="safety_contraindications", threshold=0.45, keywords=["contraindication", "avoid"]),
    ]
    passages = [
        _passage("c1", "Guideline recommendation supports treatment when clinically indicated."),
        _passage("c3", "Current guideline recommendation confirms treatment for the eligible population."),
        _passage("c2", "Review contraindications and avoid therapy when severe allergy is present."),
    ]

    check = policy.check_sufficiency(
        passages,
        query_spec=None,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )

    assert check.passed is True
    assert check.policy_decision is not None
    assert check.policy_decision.action.value == "accept"
    assert {item.facet for item in check.facet_coverage} == {"guideline_concordance", "safety_contraindications"}


def test_facet_sufficiency_routes_missing_safety_to_retrieve_more():
    policy = EvidenceSufficiencyPolicy()
    facets = [
        ClinicalFacet(name="guideline_concordance", threshold=0.50, keywords=["guideline"]),
        ClinicalFacet(name="safety_contraindications", threshold=0.75, keywords=["contraindication", "avoid"]),
    ]
    passages = [_passage("c1", "Guideline recommendation supports treatment.")]

    check = policy.check_sufficiency(
        passages,
        iteration=0,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )

    assert check.passed is False
    assert check.action_taken == "retrieve_more"
    assert "safety_contraindications" in check.missing_facets
    assert policy.get_retrieval_hints(check)["safety_search"] is True


def test_contradictions_are_preserved_in_policy_decision():
    policy = EvidenceSufficiencyPolicy()
    facets = [ClinicalFacet(name="guideline_concordance", threshold=0.30, keywords=["recommendation", "avoid"])]
    support = _passage("support", "Guideline recommendation supports use in selected adults.")
    conflict = _passage("conflict", "Avoid use because this therapy is not recommended in this population.")

    check = policy.check_sufficiency(
        [support, conflict],
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=[support, conflict],
    )

    assert check.contradiction_count >= 1
    assert check.policy_decision is not None
    assert check.policy_decision.unresolved_critical_conflicts is True
