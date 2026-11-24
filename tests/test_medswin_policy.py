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


def test_different_drugs_do_not_form_a_critical_conflict():
    policy = EvidenceSufficiencyPolicy()
    facets = [
        ClinicalFacet(
            name="guideline_concordance",
            threshold=0.30,
            keywords=["apixaban", "dabigatran", "recommended", "avoid", "fibrillation"],
        )
    ]
    support = _passage(
        "cpg-aha-afib-anticoag#0",
        "AHA guideline: apixaban is recommended over warfarin for nonvalvular atrial fibrillation with CKD.",
        source_type=SourceType.CPG,
    )
    other_drug = _passage(
        "safety-dabigatran-nsaid-ckd#0",
        "Avoid dabigatran when creatinine clearance is below 30. NSAIDs such as ibuprofen are contraindicated.",
        source_type=SourceType.SAFETY,
    )
    same_drug = _passage(
        "safety-apixaban#0",
        "Avoid apixaban when active pathological bleeding is present. Apixaban is contraindicated.",
        source_type=SourceType.SAFETY,
    )

    different = policy.check_sufficiency(
        [support, other_drug],
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=[support, other_drug],
    )
    same = policy.check_sufficiency(
        [support, same_drug],
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=[support, same_drug],
    )

    unnamed = _passage(
        "trec-generic#0",
        "A trial reports outcomes for older adults with atrial fibrillation and chronic kidney disease.",
        source_type=SourceType.LIT,
    )
    unnamed_pair = policy.check_sufficiency(
        [unnamed, other_drug],
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=[unnamed, other_drug],
    )

    assert different.contradiction_count == 0
    assert different.policy_decision.unresolved_critical_conflicts is False
    assert unnamed_pair.contradiction_count == 0
    assert same.contradiction_count >= 1
    assert same.policy_decision.unresolved_critical_conflicts is True


def test_mean_confidence_below_threshold_retrieves_more(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "SUFF_T_MEAN_CONF", 0.99)
    policy = EvidenceSufficiencyPolicy()
    facets = [
        ClinicalFacet(name="guideline_concordance", threshold=0.55, keywords=["guideline", "recommendation"]),
    ]
    passages = [
        _passage("c1", "Guideline recommendation supports treatment when clinically indicated."),
        _passage("c2", "Current guideline recommendation confirms treatment for the eligible population."),
    ]

    check = policy.check_sufficiency(
        passages,
        iteration=0,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )

    assert check.passed is False
    assert check.action_taken == "retrieve_more"
    assert "mean passage confidence" in check.policy_decision.reason.lower()
    assert check.policy_decision.retrieval_hints["mean_confidence_passed"] is False
    assert check.policy_decision.retrieval_hints["inclusion_passed"] is True


def test_passage_below_inclusion_abstains_on_the_last_loop(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "SUFF_T_INCLUSION", 0.99)
    policy = EvidenceSufficiencyPolicy()
    facets = [
        ClinicalFacet(name="guideline_concordance", threshold=0.55, keywords=["guideline", "recommendation"]),
    ]
    passages = [
        _passage("c1", "Guideline recommendation supports treatment when clinically indicated.", score=0.92),
        _passage("c2", "Current guideline recommendation confirms treatment for the eligible population.", score=0.90),
    ]

    check = policy.check_sufficiency(
        passages,
        iteration=settings.MAX_RETRIEVE_LOOPS - 1,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )

    assert check.passed is False
    assert check.action_taken == "insufficient_evidence"
    assert "inclusion threshold" in check.policy_decision.reason.lower()
    assert check.policy_decision.retrieval_hints["inclusion_passed"] is False


def test_reported_mean_keeps_a_zero_calibrated_score():
    from app.medswin.gate import EvidenceGate
    from app.schemas.enums import SourceType as SchemaSource
    from app.schemas.evidence import CandidatePassage as SchemaPassage

    passage = SchemaPassage(
        chunk_id="c1",
        doc_id="d1",
        source_type=SchemaSource.CPG,
        text="Guideline recommendation supports treatment.",
        calibrated_score=0.0,
        dense_score=0.95,
    )
    check = EvidenceGate().check([], [], [passage], contradictions=[])
    assert check.mean_confidence == 0.0
    assert check.policy_decision.retrieval_hints["mean_confidence"] == 0.0
    assert check.policy_decision.retrieval_hints["inclusion_passed"] is False
