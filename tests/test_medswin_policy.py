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


def test_a_low_caution_is_named_in_the_gate_reason():
    from app.medswin.gate import EvidenceGate
    from app.schemas.enums import ClinicalScope, PolicyAction
    from app.schemas.evidence import ContradictionPair

    passage = _passage("c1", "Use apixaban for stroke prevention.")
    pair = ContradictionPair(
        facet="guideline_concordance",
        chunk_id_a="c1",
        chunk_id_b="caution",
        severity="low",
        reason="Same-drug caution qualifies this recommendation.",
    )
    decision = EvidenceGate().decide(
        [],
        [],
        [passage],
        iteration=2,
        clinical_scope=ClinicalScope.CLINICIAN_CDS,
        contradictions=[pair],
        stop_retrieval=True,
    )
    assert decision.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert decision.reason == "An unresolved caution qualifies this recommendation."
    caution = _passage("caution", "Avoid apixaban.")
    named = EvidenceGate().decide(
        [],
        [],
        [passage, caution],
        iteration=2,
        clinical_scope=ClinicalScope.CLINICIAN_CDS,
        contradictions=[pair],
        stop_retrieval=True,
    )
    assert named.reason == "An unresolved caution qualifies apixaban."
    high = ContradictionPair(
        facet="guideline_concordance",
        chunk_id_a="c1",
        chunk_id_b="caution",
        severity="high",
        reason="The sources conflict on apixaban.",
    )
    critical = EvidenceGate().decide(
        [],
        [],
        [passage, caution],
        iteration=2,
        clinical_scope=ClinicalScope.CLINICIAN_CDS,
        contradictions=[high],
        stop_retrieval=True,
    )
    assert critical.action == PolicyAction.INSUFFICIENT_EVIDENCE
    assert critical.reason == "Unresolved evidence conflicts for apixaban need clinician review."
    medium = ContradictionPair(
        facet="guideline_concordance",
        chunk_id_a="c1",
        chunk_id_b="caution",
        severity="medium",
        reason="Same-drug safety warning conflicts with this recommendation for apixaban.",
    )
    warned = EvidenceGate().decide(
        [],
        [],
        [passage, caution],
        iteration=2,
        clinical_scope=ClinicalScope.CLINICIAN_CDS,
        contradictions=[medium],
        stop_retrieval=True,
    )
    assert warned.reason == "An unresolved caution conflicts with apixaban."
    assert "qualifies" not in warned.reason.lower()


def test_the_policy_shim_uses_the_query_text():
    policy = EvidenceSufficiencyPolicy()
    check = policy.check_sufficiency(
        [_passage("c1", "Guideline recommendation for an elderly patient.", score=0.9)],
        query="anticoagulant choice for an elderly patient",
        iteration=2,
    )
    patient = next(row for row in check.facet_coverage if row.facet == "patient_applicability")
    assert patient.required is True


def test_a_chart_row_without_a_patient_is_rejected():
    from fastapi import HTTPException

    from app.api.v1.endpoints.storage import _normalize_chunk

    try:
        _normalize_chunk(
            {"chunk_id": "chart-1", "text": "Allergy to penicillin.", "source_type": "EMR"},
            "org-1",
            SourceType.LIT,
        )
        raised = False
    except HTTPException as exc:
        raised = exc.status_code == 400
    assert raised is True


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


def test_an_empty_selection_does_not_review_the_dropped_passages():
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
    constraints = {"required_facets": [facet.model_dump() for facet in facets]}
    accepted = policy.check_sufficiency(passages, constraints=constraints, selected_passages=passages)
    empty = policy.check_sufficiency(passages, constraints=constraints, selected_passages=[])
    assert accepted.passed is True
    assert empty.passed is False


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


def test_weak_provenance_retrieves_again_before_the_last_loop():
    from app.core.config import settings

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
    passages[0].metadata = {"weak_provenance": True}

    early = policy.check_sufficiency(
        passages,
        iteration=0,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )
    late = policy.check_sufficiency(
        passages,
        iteration=settings.MAX_RETRIEVE_LOOPS - 1,
        constraints={"required_facets": [facet.model_dump() for facet in facets]},
        selected_passages=passages,
    )

    assert early.passed is False
    assert early.action_taken == "retrieve_more"
    assert "source identity" in early.policy_decision.reason
    assert late.action_taken == "insufficient_evidence"
    assert "source identity" in late.policy_decision.reason


def test_source_counts_do_not_accept_or_reject(monkeypatch):
    from app.core.config import settings

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
    constraints = {"required_facets": [facet.model_dump() for facet in facets]}
    before = policy.check_sufficiency(passages, constraints=constraints, selected_passages=passages)
    monkeypatch.setattr(settings, "SUFF_T_CPG", 9)
    monkeypatch.setattr(settings, "SUFF_T_EMR", 9)
    after = policy.check_sufficiency(passages, constraints=constraints, selected_passages=passages)

    assert before.action_taken == after.action_taken == "accept"
    assert after.kappa_cpg < before.kappa_cpg
    assert after.policy_decision.retrieval_hints["source_count_audit"] == {
        "cpg": 3,
        "emr": 0,
        "gates_action": False,
    }


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
