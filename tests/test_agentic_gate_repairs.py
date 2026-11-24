from app.medswin.ledger import (
    adjudicate_contradictions,
    detect_contradictions,
    intervention_tokens,
    same_intervention,
)
from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity, SourceType
from app.schemas.evidence import (
    CandidatePassage,
    ContradictionPair,
    EvidenceClaim,
    EvidenceGrade,
    EvidenceLedgerEntry,
)
from app.schemas.facets import ClinicalFacet
from app.scoring.coverage import infer_polarity, score_passage_facets


def _passage(chunk_id: str, text: str, source_type: SourceType = SourceType.CPG) -> CandidatePassage:
    return CandidatePassage(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        source_type=source_type,
        text=text,
        section="Recommendations",
    )


def _entry(chunk_id: str, facet: str, claim: str, polarity: EvidencePolarity, grade: float = 0.50) -> EvidenceLedgerEntry:
    evidence_grade = EvidenceGrade(score=grade)
    return EvidenceLedgerEntry(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        source_type=SourceType.CPG if facet != "safety_contraindications" else SourceType.SAFETY,
        evidence_grade=evidence_grade,
        claims=[
            EvidenceClaim(
                facet=facet,
                claim=claim,
                polarity=polarity,
                chunk_id=chunk_id,
                evidence_grade=evidence_grade,
            )
        ],
    )


def test_negated_caution_does_not_become_a_conflict():
    facet = ClinicalFacet(name="guideline_concordance", keywords=["apixaban"])
    negated = _passage("cpg-1", "The guideline reports no contraindications for selected adults.")
    delay = _passage("cpg-2", "Do not delay antibiotics when infection is confirmed.")
    warning = _passage("cpg-3", "Avoid apixaban when active bleeding is present.")

    assert infer_polarity(negated, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(delay, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(warning, facet) == EvidencePolarity.CONTRADICTS


def test_same_drug_warning_on_the_safety_facet_conflicts_with_the_guideline():
    support = _entry(
        "cpg-aha",
        "guideline_concordance",
        "Apixaban is recommended over warfarin for nonvalvular atrial fibrillation.",
        EvidencePolarity.SUPPORTS,
        grade=0.95,
    )
    same_drug = _entry(
        "safety-apixaban",
        "safety_contraindications",
        "Avoid apixaban when active pathological bleeding is present.",
        EvidencePolarity.SAFETY,
        grade=0.95,
    )
    other_drug = _entry(
        "safety-dabigatran",
        "safety_contraindications",
        "Avoid dabigatran when creatinine clearance is below 30.",
        EvidencePolarity.SAFETY,
        grade=0.95,
    )

    same = detect_contradictions([support, same_drug])
    different = detect_contradictions([support, other_drug])

    assert len(same) == 1
    assert same[0].facet == "guideline_concordance"
    assert different == []


def test_critic_must_cite_both_chunks_before_a_medium_conflict_clears():
    pair = ContradictionPair(
        facet="guideline_concordance",
        chunk_id_a="cpg-1",
        chunk_id_b="lit-1",
        severity="medium",
        reason="test",
        grade_a=0.70,
        grade_b=0.38,
    )
    note_only = adjudicate_contradictions(
        [pair.model_copy(deep=True)],
        [AgentClaimBatch(agent_id="critic", notes=["This evidence is outdated."])],
    )
    cited = adjudicate_contradictions(
        [pair.model_copy(deep=True)],
        [
            AgentClaimBatch(
                agent_id="critic",
                claims=[
                    EvidenceClaim(facet="guideline_concordance", claim="cpg-1 is outdated", polarity=EvidencePolarity.CONTRADICTS, chunk_id="cpg-1"),
                    EvidenceClaim(facet="guideline_concordance", claim="lit-1 is outdated", polarity=EvidencePolarity.CONTRADICTS, chunk_id="lit-1"),
                ],
            )
        ],
    )
    high = pair.model_copy(deep=True)
    high.severity = "high"
    high_result = adjudicate_contradictions(
        [high],
        [
            AgentClaimBatch(
                agent_id="critic",
                claims=[
                    EvidenceClaim(facet="guideline_concordance", claim="cpg-1 is outdated", polarity=EvidencePolarity.CONTRADICTS, chunk_id="cpg-1"),
                    EvidenceClaim(facet="guideline_concordance", claim="lit-1 is outdated", polarity=EvidencePolarity.CONTRADICTS, chunk_id="lit-1"),
                ],
            )
        ],
    )

    assert note_only[0].resolved is False
    assert cited[0].resolved is True
    assert high_result[0].resolved is False


def test_exact_source_match_without_a_keyword_does_not_cover_the_facet():
    facet = ClinicalFacet(name="guideline_concordance", source_policy="CPG", keywords=["apixaban"])
    bare = _passage("cpg-bare", "Older adults with atrial fibrillation were observed in clinic.")
    named = _passage("cpg-named", "Apixaban is listed for nonvalvular atrial fibrillation.")

    assert score_passage_facets(bare, [facet])["guideline_concordance"] == 0.0
    assert score_passage_facets(named, [facet])["guideline_concordance"] > 0.0


def test_drug_tokens_fold_plurals_and_ignore_april():
    assert intervention_tokens("Avoid NSAIDs") == {"nsaid"}
    assert same_intervention("Avoid NSAIDs", "NSAID is recommended") is True
    assert "april" not in intervention_tokens("Avoid therapy in April")
    assert same_intervention("Recommend lisinopril in April", "Avoid therapy in April") is False
    assert intervention_tokens("low-molecular-weight-heparin") == {"heparin"}
