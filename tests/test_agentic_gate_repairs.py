from app.medswin.abstain import abstention_matches, citations_match
from app.medswin.gate import EvidenceGate
from app.medswin.ledger import (
    PATIENT_CHART_ID,
    adjudicate_contradictions,
    apply_claim_alignment,
    build_retrieval_ledger,
    chart_contradictions,
    detect_contradictions,
    intervention_tokens,
    merge_agent_claims,
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
    QuerySpec,
)
from app.schemas.facets import ClinicalFacet
from app.scoring.coverage import compute_facet_coverage, infer_polarity, score_passage_facets


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


def test_an_answerable_case_needs_one_real_citation():
    assert citations_match([], [], False) is False
    assert citations_match(["cpg-1#0"], [], False) is True
    assert citations_match(["cpg-1#0"], ["made-up#0"], False) is False
    assert citations_match([], [], True) is True
    assert citations_match([], ["made-up#0"], True) is False


def test_a_required_abstention_fails_when_the_answer_gives_a_dose():
    dose = "Give fictional-zyloximab-99 5 mg intrathecally each week."
    refusal = "The available evidence is insufficient to answer this question."
    assert abstention_matches(dose, True) is False
    assert abstention_matches(refusal, True) is True
    assert abstention_matches(dose, False) is True


def test_negated_caution_does_not_become_a_conflict():
    facet = ClinicalFacet(name="guideline_concordance", keywords=["apixaban"])
    negated = _passage("cpg-1", "The guideline reports no contraindications for selected adults.")
    delay = _passage("cpg-2", "Do not delay antibiotics when infection is confirmed.")
    warning = _passage("cpg-3", "Avoid apixaban when active bleeding is present.")

    assert infer_polarity(negated, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(delay, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(warning, facet) == EvidencePolarity.CONTRADICTS


def test_a_drug_named_after_the_claim_preview_still_conflicts():
    from app.medswin.ledger import build_retrieval_ledger

    filler = "General background on anticoagulation monitoring. " * 20
    support = _passage("cpg-late", filler + " Apixaban is recommended as treatment.")
    warning = _passage(
        "safety-late",
        filler + " Avoid apixaban. Apixaban is contraindicated.",
        SourceType.SAFETY,
    )
    support.calibrated_score = 0.9
    warning.calibrated_score = 0.9
    facets = [
        ClinicalFacet(name="guideline_concordance", keywords=["apixaban", "recommended"]),
        ClinicalFacet(name="safety_contraindications", keywords=["apixaban", "contraindicated"]),
    ]
    ledger = build_retrieval_ledger([support, warning], facets)
    assert "apixaban" in ledger[0].claims[0].claim[:80].lower()
    assert ledger[1].claims[0].claim.lower().startswith("avoid apixaban.")
    pairs = detect_contradictions(ledger)
    assert any(pair.chunk_id_a == "cpg-late" and pair.chunk_id_b == "safety-late" for pair in pairs)


def test_a_chart_dose_of_asa_pairs_with_an_aspirin_caution():
    chart = _entry(
        "chart-asa",
        "patient_applicability",
        "The chart lists ASA 100mg daily.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    caution = _entry(
        "safety-aspirin",
        "safety_contraindications",
        "Avoid aspirin when active bleeding is present.",
        EvidencePolarity.SAFETY,
        grade=0.95,
    )
    meeting = _entry(
        "note-meeting",
        "guideline_concordance",
        "The ASA meeting used a 100mg slide.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    pairs = detect_contradictions([chart, caution])
    assert len(pairs) == 1
    assert pairs[0].chunk_id_a == "chart-asa"
    assert "aspirin" in pairs[0].reason
    assert detect_contradictions([meeting, caution]) == []


def test_an_aspirin_abbreviation_pairs_with_aspirin():
    support = _entry(
        "cpg-aspirin",
        "guideline_concordance",
        "Aspirin is recommended for secondary prevention.",
        EvidencePolarity.SUPPORTS,
        grade=0.95,
    )
    caution = _entry(
        "safety-asa",
        "safety_contraindications",
        "Avoid ASA when active bleeding is present.",
        EvidencePolarity.SAFETY,
        grade=0.95,
    )
    meeting = _entry(
        "note-asa",
        "guideline_concordance",
        "The ASA meeting reviewed the trial methods.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    pairs = detect_contradictions([support, caution])
    assert len(pairs) == 1
    assert pairs[0].chunk_id_b == "safety-asa"
    assert "aspirin" in pairs[0].reason
    assert detect_contradictions([meeting, caution]) == []


def test_a_brand_name_pairs_with_the_generic_drug():
    support = _entry(
        "cpg-apixaban",
        "guideline_concordance",
        "Apixaban is recommended for nonvalvular atrial fibrillation.",
        EvidencePolarity.SUPPORTS,
        grade=0.95,
    )
    caution = _entry(
        "safety-eliquis",
        "safety_contraindications",
        "Avoid Eliquis when active bleeding is present.",
        EvidencePolarity.SAFETY,
        grade=0.95,
    )
    pairs = detect_contradictions([support, caution])
    assert len(pairs) == 1
    assert pairs[0].chunk_id_b == "safety-eliquis"
    assert "apixaban" in pairs[0].reason


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
    assert same[0].reason == "Same-drug safety warning conflicts with this recommendation for apixaban."
    assert different == []


def test_monitoring_does_not_qualify_the_same_drug():
    facet = ClinicalFacet(name="guideline_concordance", keywords=["metformin"])
    support = _passage("cpg", "Metformin is recommended for type 2 diabetes.")
    monitor = _passage("lab", "Laboratory monitoring is recommended during metformin treatment.")
    caution = _passage("ckd", "Use metformin with caution in CKD.")
    precaution = _passage("safe", "Take precautions with metformin when the eGFR is low.")
    exceptional = _passage("trial", "Metformin has an exceptional effect in this trial.")
    support.calibrated_score = 0.9
    monitor.calibrated_score = 0.9
    ledger = build_retrieval_ledger([support, monitor], [facet])
    assert infer_polarity(monitor, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(exceptional, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(caution, facet) == EvidencePolarity.QUALIFIES
    assert infer_polarity(precaution, facet) == EvidencePolarity.QUALIFIES
    assert detect_contradictions(ledger) == []


def test_a_low_caution_is_named_when_the_gate_abstains():
    from app.medswin.abstain import insufficient_answer
    from app.schemas.enums import PolicyAction
    from app.schemas.evidence import PolicyDecision

    decision = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="A caution qualifies the recommendation.",
        contradictions=[
            ContradictionPair(
                facet="guideline_concordance",
                chunk_id_a="cpg",
                chunk_id_b="caution",
                severity="low",
                reason="Same-drug caution qualifies this recommendation.",
            )
        ],
    )
    text = insufficient_answer("Which dose?", decision)
    assert "unresolved caution" in text.lower()
    assert "high-severity" not in text.lower()
    clear = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="Missing evidence.",
    )
    assert "unresolved caution" not in insufficient_answer("Which dose?", clear).lower()
    from app.schemas.enums import SourceType

    named = insufficient_answer(
        "Which dose?",
        decision,
        passages=[
            CandidatePassage(
                chunk_id="caution",
                doc_id="doc-caution",
                source_type=SourceType.SAFETY,
                text="Avoid apixaban.",
            )
        ],
    )
    assert "apixaban" in named.lower()
    decision.unresolved_critical_conflicts = True
    decision.contradictions[0].severity = "high"
    high_text = insufficient_answer(
        "Which dose?",
        decision,
        passages=[
            CandidatePassage(
                chunk_id="caution",
                doc_id="doc-caution",
                source_type=SourceType.SAFETY,
                text="Avoid apixaban.",
            )
        ],
    )
    assert "high-severity" in high_text.lower()
    assert "apixaban" in high_text.lower()
    from app.medswin.ledger import PATIENT_CHART_ID

    chart_decision = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="Unresolved evidence conflicts for apixaban need clinician review.",
        unresolved_critical_conflicts=True,
        contradictions=[
            ContradictionPair(
                facet="guideline_concordance",
                chunk_id_a="cpg",
                chunk_id_b=PATIENT_CHART_ID,
                severity="high",
                reason=(
                    "Chart caution: Avoid apixaban. | "
                    "Same-drug safety warning conflicts with this recommendation for apixaban."
                ),
            )
        ],
    )
    chart_text = insufficient_answer("Which dose?", chart_decision)
    assert "Chart caution: Avoid apixaban." in chart_text
    medium_decision = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="An unresolved caution conflicts with apixaban.",
        contradictions=[
            ContradictionPair(
                facet="guideline_concordance",
                chunk_id_a="cpg",
                chunk_id_b="caution",
                severity="medium",
                reason="Same-drug safety warning conflicts with this recommendation for apixaban.",
            )
        ],
    )
    medium_text = insufficient_answer(
        "Which dose?",
        medium_decision,
        passages=[
            CandidatePassage(
                chunk_id="caution",
                doc_id="doc-caution",
                source_type=SourceType.SAFETY,
                text="Avoid apixaban.",
            )
        ],
    )
    assert "conflicts with apixaban" in medium_text.lower()
    assert "qualifies" not in medium_text.lower()
    mixed = PolicyDecision(
        passed=False,
        action=PolicyAction.INSUFFICIENT_EVIDENCE,
        reason="An unresolved caution conflicts with apixaban.",
        contradictions=[
            ContradictionPair(
                facet="guideline_concordance",
                chunk_id_a="cpg",
                chunk_id_b="caution",
                severity="medium",
                reason="Same-drug safety warning conflicts with this recommendation for apixaban.",
            ),
            ContradictionPair(
                facet="guideline_concordance",
                chunk_id_a="cpg",
                chunk_id_b="qualify",
                severity="low",
                reason="Same-drug caution qualifies this recommendation for metformin.",
            ),
        ],
    )
    mixed_text = insufficient_answer(
        "Which dose?",
        mixed,
        passages=[
            CandidatePassage(
                chunk_id="caution",
                doc_id="doc-caution",
                source_type=SourceType.SAFETY,
                text="Avoid apixaban.",
            ),
            CandidatePassage(
                chunk_id="qualify",
                doc_id="doc-qualify",
                source_type=SourceType.CPG,
                text="Use metformin with caution.",
            ),
        ],
    )
    assert "conflicts with apixaban" in mixed_text.lower()
    assert "qualifies metformin" in mixed_text.lower()
    assert "qualifies apixaban" not in mixed_text.lower()


def test_same_drug_caution_qualifies_the_recommendation():
    support = _entry(
        "cpg-apixaban",
        "guideline_concordance",
        "Apixaban is recommended for nonvalvular atrial fibrillation.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    caution = _entry(
        "safety-ckd",
        "safety_contraindications",
        "Use apixaban with caution in CKD.",
        EvidencePolarity.QUALIFIES,
        grade=0.80,
    )
    other = _entry(
        "safety-dabi",
        "safety_contraindications",
        "Use dabigatran with caution when clearance is low.",
        EvidencePolarity.QUALIFIES,
        grade=0.80,
    )
    same = detect_contradictions([support, caution])
    different = detect_contradictions([support, other])
    assert len(same) == 1
    assert same[0].severity == "low"
    assert same[0].chunk_id_b == "safety-ckd"
    assert same[0].reason == "Same-drug caution qualifies this recommendation for apixaban."
    assert different == []


def test_a_stored_support_label_does_not_hide_an_avoid_sentence():
    facet = ClinicalFacet(name="guideline_concordance", keywords=["apixaban"], required=True, threshold=0.70)
    caution = _passage("caut", "Avoid apixaban. Apixaban is contraindicated.")
    caution.metadata = {"polarity": "supports"}
    caution.calibrated_score = 0.95
    caution.evidence_grade_score = 0.95
    labelled = _passage("note", "Apixaban is discussed in this background note.")
    labelled.metadata = {"polarity": "irrelevant"}
    labelled.calibrated_score = 0.95
    labelled.evidence_grade_score = 0.95
    assert infer_polarity(caution, facet) == EvidencePolarity.CONTRADICTS
    assert infer_polarity(labelled, facet) == EvidencePolarity.IRRELEVANT
    caution_row = compute_facet_coverage([facet], build_retrieval_ledger([caution], [facet]))[0]
    labelled_row = compute_facet_coverage([facet], build_retrieval_ledger([labelled], [facet]))[0]
    assert caution_row.coverage_probability == 0.0
    assert caution_row.supporting_chunk_ids == []
    assert labelled_row.coverage_probability == 0.0
    assert labelled_row.supporting_chunk_ids == []


def test_a_precaution_alone_does_not_cover_the_facet():
    facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban"],
        required=True,
        threshold=0.70,
    )
    precaution = _passage(
        "safety-ckd",
        "Use apixaban with caution in CKD.",
        source_type=SourceType.CPG,
    )
    precaution.calibrated_score = 0.95
    precaution.evidence_grade_score = 0.95
    recommendation = _passage(
        "cpg-apixaban",
        "Apixaban is recommended for nonvalvular atrial fibrillation.",
        source_type=SourceType.CPG,
    )
    recommendation.calibrated_score = 0.95
    recommendation.evidence_grade_score = 0.95
    caution_only = compute_facet_coverage([facet], build_retrieval_ledger([precaution], [facet]))
    supported = compute_facet_coverage([facet], build_retrieval_ledger([recommendation], [facet]))
    assert caution_only[0].coverage_probability == 0.0
    assert caution_only[0].supporting_chunk_ids == []
    assert "safety-ckd" in caution_only[0].contradicting_chunk_ids
    assert supported[0].coverage_probability > 0.0
    assert supported[0].supporting_chunk_ids == ["cpg-apixaban"]


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


def test_one_keyword_on_a_matching_source_does_not_clear_the_facet():
    weak_facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban"],
        source_policy="CPG",
        required=True,
        threshold=0.70,
    )
    weak = _passage("weak", "See apixaban in the appendix.")
    weak.calibrated_score = 1.0
    weak.evidence_grade_score = 1.0
    weak_row = compute_facet_coverage([weak_facet], build_retrieval_ledger([weak], [weak_facet]))[0]
    assert weak_row.coverage_probability < 0.70
    assert weak_row.status != "satisfied"

    strong_facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban", "first-line"],
        source_policy="CPG",
        required=True,
        threshold=0.70,
    )
    strong = _passage("strong", "Apixaban is recommended as first-line therapy for atrial fibrillation.")
    strong.calibrated_score = 1.0
    strong.evidence_grade_score = 1.0
    strong_row = compute_facet_coverage([strong_facet], build_retrieval_ledger([strong], [strong_facet]))[0]
    assert strong_row.status == "satisfied"


def test_a_missing_calibrated_score_fails_inclusion():
    passage = CandidatePassage(
        chunk_id="c1",
        doc_id="d1",
        source_type=SourceType.CPG,
        text="Guideline recommendation supports treatment.",
        fusion_score=0.95,
        rerank_score=0.95,
        dense_score=0.95,
    )
    check = EvidenceGate().check([], [], [passage], contradictions=[])
    assert check.policy_decision.retrieval_hints["inclusion_passed"] is False
    assert check.mean_confidence == 0.0


def test_an_expanded_search_keeps_the_synonym_off_the_chart_history():
    from app.retrieval.hints import expand_query
    from app.retrieval.lexical import LexicalRetriever

    context = "Avoid apixaban. " + " ".join(f"history{i}" for i in range(20))
    query = f"Which metformin dose?\n\nPatient retrieval context:\n{context}"
    hints = {"expand_synonyms": True, "missing_facets": ["safety_contraindications"]}
    expanded = expand_query(query, hints)
    terms = LexicalRetriever._lexical_terms(expanded)
    focus = LexicalRetriever.focus_query(expanded)
    assert "biguanide" in terms
    assert "metformin" in terms
    assert "apixaban" in terms
    assert "history19" not in terms
    assert "history19" not in focus
    assert "biguanide" in focus.lower()
    assert expand_query(expanded, hints) == expanded


def test_a_generic_facet_word_does_not_outrank_the_question_drug():
    from app.retrieval.hints import expand_query
    from app.retrieval.lexical import BM25Okapi, LexicalRetriever, _matching_indexes

    expanded = expand_query(
        "Which metformin dose?",
        {"expand_synonyms": True, "missing_facets": ["safety_contraindications"]},
    )
    terms = LexicalRetriever._lexical_terms(expanded)
    assert "safety" not in terms
    assert "contraindications" in terms
    filler = LexicalRetriever._bm25_tokens("This safety note discusses safety equipment in the ward.")
    drug = LexicalRetriever._bm25_tokens("Metformin is first-line treatment.")
    caution = LexicalRetriever._bm25_tokens("Avoid apixaban when bleeding is present.")
    docs = [filler, drug, caution]
    docs += [LexicalRetriever._bm25_tokens(f"Unrelated topic number {i} about diet.") for i in range(8)]
    order = _matching_indexes(BM25Okapi(docs), terms, 3)
    assert 0 not in order
    assert order[0] == 1
    assert 2 in order


def test_an_expanded_caution_phrase_does_not_outrank_the_question_drug():
    from app.retrieval.hints import expand_query
    from app.retrieval.lexical import BM25Okapi, LexicalRetriever, _matching_indexes

    context = "Avoid apixaban. " + " ".join(f"history{i}" for i in range(20))
    query = f"Which metformin dose?\n\nPatient retrieval context:\n{context}"
    expanded = expand_query(
        query,
        {"expand_synonyms": True, "missing_facets": ["safety_contraindications"]},
    )
    terms = LexicalRetriever._lexical_terms(expanded)
    assert "do" not in terms
    assert "recommended" not in terms
    filler = LexicalRetriever._bm25_tokens("Do not use this form. It is not recommended for the note.")
    drug = LexicalRetriever._bm25_tokens("Metformin is first-line treatment.")
    caution = LexicalRetriever._bm25_tokens("Avoid apixaban when bleeding is present.")
    order = _matching_indexes(BM25Okapi([filler, drug, caution]), terms, 3)
    assert 0 not in order
    assert order[0] == 2


def test_a_second_expansion_does_not_change_the_query():
    from app.retrieval.hints import expand_query

    hints = {"expand_synonyms": True, "missing_facets": ["safety_contraindications"]}
    once = expand_query("metformin renal dose", hints)
    twice = expand_query(once, hints)
    assert "biguanide" in once
    assert "kidney" in once
    assert "avoid" in once
    assert once.count("biguanide") == 1
    assert once.count("avoid") == 1
    assert twice == once


def test_a_duplicate_page_abstains_at_the_real_iteration():
    from app.medswin.orchestrator import retrieve_stall_action

    assert retrieve_stall_action(
        iteration=1,
        max_loops=3,
        hints={"missing_facets": ["safety_contraindications"], "focus_source": "SAFETY"},
        next_hints={"missing_facets": ["safety_contraindications"], "focus_source": "SAFETY"},
        wants_more=True,
    ) == "abstain"
    assert retrieve_stall_action(
        iteration=1,
        max_loops=3,
        hints={"missing_facets": ["safety_contraindications"], "focus_source": "SAFETY"},
        next_hints={"missing_facets": ["patient_applicability"], "focus_source": "EMR"},
        wants_more=True,
    ) == "continue"


def test_a_lost_session_insert_does_not_keep_another_users_id():
    import asyncio

    from pymongo.errors import DuplicateKeyError

    from app.medswin.orchestrator import _resume_or_create_session

    created = []

    class Sessions:
        def __init__(self):
            self.reads = 0

        async def update_last_active(self, session_id, org_id, user_id=None):
            return False

        async def get_by_id(self, session_id, org_id):
            self.reads += 1
            if self.reads == 1:
                return None
            return {
                "session_id": session_id,
                "org_id": org_id,
                "user_id": "other-clinician",
                "patient_id": None,
            }

        async def create(self, session, org_id):
            if session.session_id == "sess-race":
                raise DuplicateKeyError("duplicate")
            created.append(session.session_id)
            return {"session_id": session.session_id}

    new_id = asyncio.run(_resume_or_create_session(Sessions(), "sess-race", "dr-chen", "org-1"))
    assert new_id != "sess-race"
    assert created == [new_id]


def test_a_lost_session_insert_keeps_the_id_when_this_user_owns_it():
    import asyncio

    from pymongo.errors import DuplicateKeyError

    from app.medswin.orchestrator import _resume_or_create_session

    class Sessions:
        def __init__(self):
            self.reads = 0

        async def update_last_active(self, session_id, org_id, user_id=None):
            return False

        async def get_by_id(self, session_id, org_id):
            self.reads += 1
            if self.reads == 1:
                return None
            return {
                "session_id": session_id,
                "org_id": org_id,
                "user_id": "dr-chen",
                "patient_id": None,
            }

        async def create(self, session, org_id):
            raise DuplicateKeyError("duplicate")

    kept = asyncio.run(_resume_or_create_session(Sessions(), "sess-race", "dr-chen", "org-1"))
    assert kept == "sess-race"


def test_a_session_trace_list_stays_with_the_owner():
    import asyncio

    from app.repositories.traces import TraceRepository

    class _Cursor:
        def __init__(self, query):
            self.query = query

        def sort(self, *_args):
            return self

        def limit(self, _count):
            return self

        async def to_list(self, length=None):
            return [{"trace_id": "trace-1", "user_id": self.query["user_id"]}]

    class _Store:
        def __init__(self):
            self.queries = []

        def find(self, query):
            self.queries.append(query)
            return _Cursor(query)

    store = _Store()
    repo = TraceRepository()
    repo._db = {"traces": store}
    rows = asyncio.run(repo.get_by_session_id("sess-1", "org-a", "dr-chen"))
    empty = asyncio.run(repo.get_by_session_id("sess-1", "org-a", ""))
    assert rows == [{"trace_id": "trace-1", "user_id": "dr-chen"}]
    assert empty == []
    assert store.queries == [{"session_id": "sess-1", "org_id": "org-a", "user_id": "dr-chen"}]


def test_a_chart_update_keeps_the_patient_and_the_identity_fields():
    import asyncio

    from app.repositories.chunks import ChunkRepository
    from app.repositories.documents import DocumentRepository

    class _Result:
        matched_count = 1

    class _Store:
        def __init__(self):
            self.calls = []

        async def update_one(self, query, update):
            self.calls.append((query, update))
            return _Result()

    chunk_store = _Store()
    chunks = ChunkRepository()
    chunks._db = {"chunks": chunk_store}
    saved = asyncio.run(chunks.update("chunk-1", "org-a", {"text": "held"}))
    moved = asyncio.run(chunks.update("chunk-1", "org-a", {"patient_id": "pt-2"}))
    nested = asyncio.run(chunks.update("chunk-1", "org-a", {"metadata.note": "x"}))
    document_store = _Store()
    documents = DocumentRepository()
    documents._db = {"documents": document_store}
    retitled = asyncio.run(documents.update("doc-1", "org-a", {"title": "held"}))
    relabelled = asyncio.run(documents.update("doc-1", "org-a", {"source_type": "LIT"}))
    assert saved is True
    assert moved is False
    assert nested is False
    assert retitled is True
    assert relabelled is False
    assert chunk_store.calls == [
        ({"chunk_id": "chunk-1", "org_id": "org-a"}, {"$set": {"text": "held"}})
    ]
    assert document_store.calls == [
        ({"doc_id": "doc-1", "org_id": "org-a"}, {"$set": {"title": "held"}})
    ]


def test_a_failed_chunk_file_keeps_the_chart_text_off_disk(tmp_path, monkeypatch, caplog):
    import logging

    import pymongo

    from app.services import storage as storage_mod

    monkeypatch.setattr(storage_mod.settings, "DATA_DIR", str(tmp_path))

    class _Coll:
        def bulk_write(self, _operations, ordered=False):
            raise pymongo.errors.BulkWriteError(
                {
                    "writeErrors": [
                        {
                            "code": 11000,
                            "errmsg": "patient phone 555-234-5678",
                            "op": {"text": "Metformin dose and phone 555-234-5678"},
                        }
                    ]
                }
            )

    class _DB:
        def __getitem__(self, _name):
            return _Coll()

    class _Client:
        def __init__(self, _url):
            return None

        def __getitem__(self, _name):
            return _DB()

        def close(self):
            return None

    monkeypatch.setattr(storage_mod, "MongoClient", _Client)
    with caplog.at_level(logging.ERROR):
        result = storage_mod.StorageService()._store_chunks_sync(
            [
                {
                    "chunk_id": "c1",
                    "org_id": "org-a",
                    "doc_id": "d1",
                    "patient_id": "pt-1",
                    "text": "Metformin dose and phone 555-234-5678",
                }
            ],
            "chunks",
            10,
        )
    body = (tmp_path / "failed_chunks.json").read_text()
    assert result["failed_count"] == 1
    assert "c1" in body
    assert "555-234-5678" not in body
    assert "Metformin" not in body
    assert "pt-1" not in body
    assert "555-234-5678" not in caplog.text
    assert "Metformin" not in caplog.text
    assert "BulkWriteError" in caplog.text


def test_a_chart_replace_keeps_the_patient():
    import asyncio

    from app.models.medswin import Chunk, Document
    from app.repositories.chunks import ChunkRepository
    from app.repositories.documents import DocumentRepository

    class _Result:
        upserted_id = None

    class _Store:
        def __init__(self, existing):
            self.existing = existing
            self.replaces = []

        async def find_one(self, _query, _projection=None):
            return self.existing

        async def replace_one(self, query, data, upsert=False):
            self.replaces.append((query, data.get("patient_id")))
            return _Result()

    chunk_store = _Store({"patient_id": "pt-1", "source_type": "EMR", "doc_id": "doc-1"})
    chunks = ChunkRepository()
    chunks._db = {"chunks": chunk_store}
    moved = Chunk(
        chunk_id="c1",
        doc_id="doc-1",
        source_type=SourceType.EMR,
        text="moved",
        patient_id="pt-2",
        org_id="org-a",
    )
    kept = moved.model_copy(update={"patient_id": "pt-1", "text": "same patient"})
    rejected = asyncio.run(chunks.create(moved, "org-a"))
    saved = asyncio.run(chunks.create(kept, "org-a"))
    document_store = _Store({"patient_id": "pt-1", "source_type": "EMR"})
    documents = DocumentRepository()
    documents._db = {"documents": document_store}
    moved_doc = Document(
        doc_id="doc-1",
        source_type=SourceType.LIT,
        title="moved",
        patient_id="pt-1",
        org_id="org-a",
    )
    relabelled = asyncio.run(documents.create(moved_doc, "org-a"))
    assert rejected["rejected"] is True
    assert saved["rejected"] is False
    assert relabelled["rejected"] is True
    assert chunk_store.replaces == [({"org_id": "org-a", "chunk_id": "c1"}, "pt-1")]
    assert document_store.replaces == []

    class _Cursor:
        async def to_list(self, length=None):
            return [{"chunk_id": "c1", "patient_id": "pt-1", "source_type": "EMR", "doc_id": "doc-1"}]

    class _Batch:
        def __init__(self):
            self.operations = []

        def find(self, _query, _projection=None):
            return _Cursor()

        async def bulk_write(self, operations, ordered=False):
            self.operations.extend(operations)

            class _Bulk:
                upserted_ids = {}

            return _Bulk()

    batch = _Batch()
    many = ChunkRepository()
    many._db = {"chunks": batch}
    fresh = kept.model_copy(update={"chunk_id": "c2", "text": "new"})
    rows = asyncio.run(many.create_many([moved, fresh], "org-a"))
    assert [row["rejected"] for row in rows] == [True, False]
    assert len(batch.operations) == 1


def test_a_trace_update_keeps_the_owner_and_the_identity_fields(monkeypatch):
    import asyncio

    from app.repositories.traces import TraceRepository

    monkeypatch.setattr("app.repositories.traces.settings.LOG_REDACT_PHI", True)

    class _Result:
        matched_count = 1

    class _Store:
        def __init__(self):
            self.calls = []

        async def update_one(self, query, update):
            self.calls.append((query, update))
            return _Result()

    store = _Store()
    repo = TraceRepository()
    repo._db = {"traces": store}
    saved = asyncio.run(repo.update("trace-1", "org-a", {"final_answer": "held"}, "dr-chen"))
    stolen = asyncio.run(repo.update("trace-1", "org-a", {"user_id": "dr-patel"}, "dr-chen"))
    nested = asyncio.run(repo.update("trace-1", "org-a", {"a.b": "c"}, "dr-chen"))
    assert saved is True
    assert stolen is False
    assert nested is False
    assert store.calls == [
        (
            {"trace_id": "trace-1", "org_id": "org-a", "user_id": "dr-chen"},
            {"$set": {"final_answer": "held"}},
        )
    ]
    phone = asyncio.run(
        repo.update("trace-1", "org-a", {"final_answer": "Call 555-234-5678. Continue metformin."}, "dr-chen")
    )
    assert phone is True
    written = store.calls[-1][1]["$set"]["final_answer"]
    assert "555-234-5678" not in written
    assert "metformin" in written


def test_session_metadata_merges_for_the_owner_only():
    import asyncio

    from app.repositories.sessions import SessionRepository

    class _Result:
        matched_count = 1
        deleted_count = 1

    class _Store:
        def __init__(self):
            self.calls = []

        async def update_one(self, query, update):
            self.calls.append(("update", query, update))
            return _Result()

        async def delete_one(self, query):
            self.calls.append(("delete", query))
            return _Result()

    store = _Store()
    repo = SessionRepository()
    repo._db = {"sessions": store}
    saved = asyncio.run(repo.update_metadata("sess-1", "org-a", {"note": "a", "flag": "b"}, "dr-chen"))
    rejected = asyncio.run(repo.update_metadata("sess-1", "org-a", {"a.b": "c"}, "dr-chen"))
    removed = asyncio.run(repo.delete("sess-1", "org-a", "dr-chen"))
    assert saved is True
    assert rejected is False
    assert removed is True
    assert store.calls[0][1]["user_id"] == "dr-chen"
    assert store.calls[0][2] == {"$set": {"metadata.note": "a", "metadata.flag": "b"}}
    noted = asyncio.run(
        repo.update_metadata("sess-1", "org-a", {"note": "Call 555-234-5678. Creatinine is 1.4."}, "dr-chen")
    )
    assert noted is True
    assert "555-234-5678" not in store.calls[-1][2]["$set"]["metadata.note"]
    assert "Creatinine is 1.4." in store.calls[-1][2]["$set"]["metadata.note"]
    assert store.calls[1][1]["user_id"] == "dr-chen"
    assert len(store.calls) == 3


def test_another_users_session_id_is_not_reused():
    import asyncio

    from app.medswin.orchestrator import _resume_or_create_session

    created = []

    class Sessions:
        async def update_last_active(self, session_id, org_id, user_id=None):
            return False

        async def get_by_id(self, session_id, org_id):
            return {"session_id": session_id, "org_id": org_id, "user_id": "other-clinician"}

        async def create(self, session, org_id):
            created.append(session.session_id)
            return {"session_id": session.session_id}

    new_id = asyncio.run(_resume_or_create_session(Sessions(), "sess-owned", "dr-chen", "org-1"))
    assert new_id != "sess-owned"
    assert created == [new_id]


def test_faiss_and_hnsw_distances_share_one_cosine_scale():
    from app.indexes.hybrid import ann_similarity

    assert ann_similarity(0.2, "HNSW") == 0.8
    assert abs(ann_similarity(0.4, "FAISS") - 0.8) < 1e-9


def test_a_keyword_does_not_match_inside_another_word():
    from app.scoring.coverage import keyword_in_text, score_passage_facets

    assert keyword_in_text("age", "management of atrial fibrillation") is False
    assert keyword_in_text("lab", "available evidence") is False
    assert keyword_in_text("apixaban", "Start apixaban 5 mg.") is True
    assert keyword_in_text("first-line", "First-line guideline recommends amoxicillin.") is True
    facet = ClinicalFacet(name="patient_applicability", keywords=["age"], source_policy="EMR")
    passage = _passage("emr-1", "Management of atrial fibrillation in clinic.", source_type=SourceType.EMR)
    assert score_passage_facets(passage, [facet])["patient_applicability"] == 0.0


def test_a_two_term_caution_survives_packing():
    from app.scoring.utility import select_bundle

    strong = CandidatePassage(
        chunk_id="strong",
        doc_id="strong",
        source_type=SourceType.CPG,
        text="Guideline recommendation for apixaban.",
        calibrated_score=0.90,
        fusion_score=0.95,
        token_count=80,
    )
    caution = CandidatePassage(
        chunk_id="caution",
        doc_id="caution",
        source_type=SourceType.SAFETY,
        text="Avoid apixaban. Apixaban is contraindicated.",
        calibrated_score=0.80,
        fusion_score=0.60,
        contradiction_score=0.32,
        token_count=20,
    )
    selected = select_bundle([strong, caution], token_budget=200, max_chunks=1)
    assert "caution" in {item.chunk_id for item in selected}


def test_a_second_drug_caution_stays_in_the_bundle():
    from app.scoring.utility import select_bundle

    strong = CandidatePassage(
        chunk_id="strong",
        doc_id="strong",
        source_type=SourceType.CPG,
        text="Guideline recommendation for apixaban.",
        calibrated_score=0.90,
        fusion_score=0.95,
        token_count=80,
    )
    apixaban = CandidatePassage(
        chunk_id="apixaban",
        doc_id="apixaban",
        source_type=SourceType.SAFETY,
        text="Avoid apixaban. Apixaban is contraindicated.",
        calibrated_score=0.80,
        fusion_score=0.60,
        contradiction_score=0.32,
        token_count=20,
    )
    warfarin = CandidatePassage(
        chunk_id="warfarin",
        doc_id="warfarin",
        source_type=SourceType.SAFETY,
        text="Avoid warfarin. Warfarin is contraindicated.",
        calibrated_score=0.80,
        fusion_score=0.60,
        contradiction_score=0.32,
        token_count=20,
    )
    repeat = CandidatePassage(
        chunk_id="apixaban-repeat",
        doc_id="apixaban-repeat",
        source_type=SourceType.SAFETY,
        text="Apixaban is not recommended.",
        calibrated_score=0.80,
        fusion_score=0.59,
        contradiction_score=0.32,
        token_count=20,
    )
    selected = select_bundle([strong, apixaban, warfarin, repeat], token_budget=200, max_chunks=1)
    kept = {item.chunk_id for item in selected}
    assert "apixaban" in kept
    assert "warfarin" in kept
    assert "apixaban-repeat" not in kept


def test_a_required_chart_replaces_a_filler_passage():
    from app.scoring.utility import select_bundle

    facet = ClinicalFacet(
        name="patient_applicability",
        required=True,
        source_policy="EMR",
        keywords=["patient"],
    )
    strong = CandidatePassage(
        chunk_id="strong",
        doc_id="strong",
        source_type=SourceType.CPG,
        text="A guideline note about follow-up.",
        calibrated_score=0.90,
        fusion_score=0.95,
        token_count=50,
    )
    filler = CandidatePassage(
        chunk_id="filler",
        doc_id="filler",
        source_type=SourceType.LIT,
        text="A literature note about follow-up.",
        calibrated_score=0.90,
        fusion_score=0.90,
        token_count=40,
    )
    chart = CandidatePassage(
        chunk_id="chart",
        doc_id="chart",
        source_type=SourceType.EMR,
        text="Patient history shows renal disease.",
        calibrated_score=0.80,
        fusion_score=0.40,
        token_count=30,
    )
    selected = select_bundle([strong, filler, chart], facets=[facet], token_budget=90, max_chunks=2)
    kept = {item.chunk_id for item in selected}
    assert "chart" in kept
    assert "strong" in kept
    assert "filler" not in kept


def test_a_new_drug_caution_replaces_a_filler_passage():
    from app.scoring.utility import select_bundle

    recommendation = CandidatePassage(
        chunk_id="rec",
        doc_id="rec",
        source_type=SourceType.CPG,
        text="Apixaban is recommended.",
        calibrated_score=0.90,
        fusion_score=0.95,
        token_count=60,
    )
    filler = CandidatePassage(
        chunk_id="filler",
        doc_id="filler",
        source_type=SourceType.LIT,
        text="A general note about clinic follow-up.",
        calibrated_score=0.80,
        fusion_score=0.70,
        token_count=30,
    )
    caution = CandidatePassage(
        chunk_id="warfarin",
        doc_id="warfarin",
        source_type=SourceType.SAFETY,
        text="Avoid warfarin. Warfarin is contraindicated.",
        calibrated_score=0.80,
        fusion_score=0.40,
        contradiction_score=0.32,
        token_count=30,
    )
    selected = select_bundle([recommendation, filler, caution], token_budget=90, max_chunks=2)
    kept = [item.chunk_id for item in selected]
    assert "rec" in kept
    assert "warfarin" in kept
    assert "filler" not in kept


def test_a_new_drug_caution_replaces_a_filler_at_the_passage_cap():
    from app.core.config import settings
    from app.scoring.utility import select_bundle

    passages = [
        CandidatePassage(
            chunk_id="rec",
            doc_id="rec",
            source_type=SourceType.CPG,
            text="Apixaban is recommended.",
            calibrated_score=0.90,
            fusion_score=0.99,
            token_count=10,
        )
    ]
    passages.extend(
        CandidatePassage(
            chunk_id=f"filler-{index}",
            doc_id=f"filler-{index}",
            source_type=SourceType.LIT,
            text=f"A general clinic note {index}.",
            calibrated_score=0.80,
            fusion_score=0.90 - (index * 0.01),
            token_count=10,
        )
        for index in range(settings.MMR_MAX_EVIDENCE_CHUNKS)
    )
    passages.append(
        CandidatePassage(
            chunk_id="warfarin",
            doc_id="warfarin",
            source_type=SourceType.SAFETY,
            text="Avoid warfarin. Warfarin is contraindicated.",
            calibrated_score=0.80,
            fusion_score=0.20,
            contradiction_score=0.32,
            token_count=10,
        )
    )
    selected = select_bundle(passages, token_budget=1000, max_chunks=settings.MMR_MAX_EVIDENCE_CHUNKS)
    kept = {item.chunk_id for item in selected}
    assert len(selected) <= settings.MMR_MAX_EVIDENCE_CHUNKS
    assert "rec" in kept
    assert "warfarin" in kept
    assert "filler-8" not in kept


def test_a_passage_below_inclusion_is_not_packed():
    from app.scoring.utility import select_bundle

    strong = CandidatePassage(
        chunk_id="strong",
        doc_id="strong",
        source_type=SourceType.CPG,
        text="Guideline recommendation for apixaban.",
        calibrated_score=0.90,
        fusion_score=0.90,
        token_count=40,
    )
    weak = CandidatePassage(
        chunk_id="weak",
        doc_id="weak",
        source_type=SourceType.LIT,
        text="Short note.",
        calibrated_score=0.30,
        fusion_score=0.45,
        token_count=3,
    )
    selected = select_bundle([weak, strong], token_budget=400, max_chunks=4)
    assert [item.chunk_id for item in selected] == ["strong"]


def test_a_rerank_miss_keeps_facet_relevance_and_still_fails_inclusion():
    from app.medswin.ledger import build_retrieval_ledger
    from app.scoring.fusion import compute_fusion_scores

    passage = CandidatePassage(
        chunk_id="cpg-dense",
        doc_id="doc-dense",
        source_type=SourceType.CPG,
        text="Guideline recommendation for apixaban.",
        dense_score=0.82,
        lexical_score=0.40,
    )
    compute_fusion_scores([passage])
    facets = [
        ClinicalFacet(
            name="guideline_concordance",
            required=True,
            threshold=0.55,
            keywords=["guideline", "recommendation"],
            source_policy="CPG",
        )
    ]
    ledger = build_retrieval_ledger([passage], facets)
    assert ledger[0].calibrated_relevance > 0
    assert passage.calibrated_score is None
    check = EvidenceGate().check(ledger, facets, [passage], iteration=2)
    assert check.passed is False


def test_mixed_retrieval_probes_the_guideline_source():
    import asyncio

    import numpy as np

    from app.retrieval.hybrid import SOURCE_BALANCE_PROBES, HybridRetriever

    assert SourceType.CPG in SOURCE_BALANCE_PROBES
    seen = []

    async def _retrieve(*args, **kwargs):
        seen.append(args[3] if len(args) > 3 else kwargs.get("source_type"))
        return []

    retriever = HybridRetriever(reranker_client=None)
    retriever.dense.retrieve = _retrieve
    retriever.lexical.retrieve = _retrieve
    asyncio.run(retriever.retrieve(query="apixaban dose", query_embedding=np.zeros(4), org_id="org-1"))
    assert SourceType.CPG in seen


def test_one_document_bound_is_its_coverage_probability():
    from app.core.config import settings
    from app.scoring.coverage import compute_facet_coverage

    assert settings.SUFF_LCB_MARGIN == 0.08
    grade = EvidenceGrade(score=1.0)
    entry = EvidenceLedgerEntry(
        chunk_id="one",
        doc_id="one",
        source_type=SourceType.CPG,
        calibrated_relevance=1.0,
        evidence_grade=grade,
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Guideline recommendation.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="one",
                confidence=0.80,
                calibrated_relevance=1.0,
                evidence_grade=grade,
            )
        ],
    )
    facet = ClinicalFacet(name="guideline_concordance", required=True, threshold=0.78)
    row = compute_facet_coverage([facet], [entry])[0]
    assert row.coverage_probability == 0.80
    assert row.lower_confidence_bound == 0.80
    assert row.status == "satisfied"


def test_a_contradiction_does_not_raise_coverage():
    from app.scoring.coverage import compute_facet_coverage

    grade = EvidenceGrade(score=1.0)
    entry = EvidenceLedgerEntry(
        chunk_id="clash",
        doc_id="clash",
        source_type=SourceType.LIT,
        calibrated_relevance=1.0,
        evidence_grade=grade,
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="This recommendation is outdated.",
                polarity=EvidencePolarity.CONTRADICTS,
                chunk_id="clash",
                confidence=0.90,
                calibrated_relevance=1.0,
                evidence_grade=grade,
            )
        ],
    )
    facet = ClinicalFacet(name="guideline_concordance", required=True, threshold=0.55)
    row = compute_facet_coverage([facet], [entry])[0]
    assert row.coverage_probability == 0.0
    assert row.supporting_chunk_ids == []
    assert row.status == "contradicted"


def test_a_contradiction_lowers_marginal_utility():
    from app.core.config import settings
    from app.scoring.utility import marginal_utility

    assert settings.CONTRADICTION_PENALTY_WEIGHT == 0.40
    plain = CandidatePassage(
        chunk_id="plain",
        doc_id="plain",
        source_type=SourceType.CPG,
        text="Guideline recommendation.",
        fusion_score=0.80,
        calibrated_score=0.80,
        contradiction_score=0.0,
    )
    clash = plain.model_copy(update={"chunk_id": "clash", "doc_id": "clash", "contradiction_score": 0.50})
    assert marginal_utility(clash, [], None) < marginal_utility(plain, [], None)


def test_outpatient_does_not_force_the_chart_facet():
    from app.medswin.normalize import QueryNormalizer

    facets = QueryNormalizer.build_facets("outpatient management of nonelderly adults", QuerySpec(canonical_terms=["management"]), {}, None)
    by_name = {facet.name: facet for facet in facets}
    assert by_name["patient_applicability"].required is False


def test_live_report_status_does_not_pass_when_the_answerable_query_abstains():
    import importlib.util

    path = "/Users/liamle/Downloads/medswin/scripts/verify_live_fullstack.py"
    spec = importlib.util.spec_from_file_location("verify_live_fullstack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.live_report_status(
        two_stage_passed=False,
        full_passed=False,
        abstained=True,
        abstain_action="insufficient_evidence",
        card_indicator="warning",
    ) == "ABSTAINED"
    assert module.live_report_status(
        two_stage_passed=True,
        full_passed=True,
        abstained=True,
        abstain_action="insufficient_evidence",
        card_indicator="info",
    ) == "PASSED"
    assert module.live_report_status(
        two_stage_passed=True,
        full_passed=True,
        abstained=False,
        abstain_action="accept",
        card_indicator="info",
    ) == "FAILED"
    assert module.LIVE_CHAT_PATHS[0].endswith("/naive/chat")
    assert module.compare_covers_the_three_pipelines(
        {
            "results": {"naive": {}, "two-stage": {}, "full": {}},
            "adjacent_diffs": {"naive_to_two_stage": {}, "two_stage_to_full": {}},
        }
    )
    assert module.compare_covers_the_three_pipelines(
        {"results": {"naive": {}, "two-stage": {}}, "adjacent_diffs": {"naive_to_two_stage": {}}}
    ) is False


def test_a_negated_caution_does_not_open_the_safety_facet():
    negated = _passage("cpg-neg", "The guideline reports no contraindications for selected adults.")
    delay = _passage("cpg-delay", "Do not delay antibiotics when infection is confirmed.")
    direct = _passage("cpg-do", "Do not use metformin when the eGFR is low.")
    facets = [ClinicalFacet(name="safety_contraindications", keywords=["contraindicated"], threshold=0.70)]
    assert score_passage_facets(negated, facets)["safety_contraindications"] == 0.0
    assert score_passage_facets(delay, facets)["safety_contraindications"] == 0.0
    assert score_passage_facets(direct, facets)["safety_contraindications"] > 0.0


def test_a_dose_sentence_does_not_open_the_safety_facet():
    dose = _passage("cpg-dose", "The metformin dose is 500 mg each morning.")
    caution = _passage("cpg-caution", "Avoid metformin when the eGFR is low.")
    facets = [ClinicalFacet(name="safety_contraindications", keywords=["contraindicated"], threshold=0.70)]
    dose_scores = score_passage_facets(dose, facets)
    caution_scores = score_passage_facets(caution, facets)
    assert dose_scores["safety_contraindications"] == 0.0
    assert caution_scores["safety_contraindications"] > 0.0
    assert "safety_contraindications" not in build_retrieval_ledger([dose], facets)[0].facets


def test_a_stored_facet_score_cannot_cover_text_that_lacks_the_facet():
    passage = _passage("cpg-1", "Metformin is first-line for type 2 diabetes.")
    passage.metadata = {"facet_scores": {"safety_contraindications": 1.0, "guideline_concordance": 0.95}}
    facets = [
        ClinicalFacet(name="guideline_concordance", keywords=["metformin"], threshold=0.70),
        ClinicalFacet(name="safety_contraindications", keywords=["contraindicated"], threshold=0.70),
    ]
    scores = score_passage_facets(passage, facets)
    plain = _passage("cpg-plain", "Metformin is first-line for type 2 diabetes.")
    plain_score = score_passage_facets(plain, facets)["guideline_concordance"]
    assert scores["safety_contraindications"] == 0.0
    assert scores["guideline_concordance"] == plain_score
    assert scores["guideline_concordance"] < 0.95
    ledger = build_retrieval_ledger([passage], facets)
    assert "safety_contraindications" not in ledger[0].facets
    assert "guideline_concordance" in ledger[0].facets


def test_a_stored_facet_score_does_not_satisfy_a_weak_sentence():
    facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban"],
        required=True,
        threshold=0.70,
    )
    weak = _passage("weak", "See apixaban in the appendix.")
    weak.metadata = {"facet_scores": {"guideline_concordance": 1.0}}
    weak.calibrated_score = 0.95
    weak.evidence_grade_score = 0.95
    row = compute_facet_coverage([facet], build_retrieval_ledger([weak], [facet]))[0]
    assert row.coverage_probability < facet.threshold
    assert row.status != "satisfied"


def test_a_specialist_confidence_does_not_satisfy_a_weak_sentence():
    facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban"],
        required=True,
        threshold=0.70,
    )
    weak = _passage("weak", "See apixaban in the appendix.")
    weak.calibrated_score = 0.95
    weak.evidence_grade_score = 0.95
    score_passage_facets(weak, [facet])
    text_score = weak.facet_scores["guideline_concordance"]
    ledger = build_retrieval_ledger([weak], [facet])
    batch = AgentClaimBatch(
        agent_id="guideline",
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is indicated.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="weak",
                confidence=1.0,
                calibrated_relevance=1.0,
                evidence_grade=EvidenceGrade(score=1.0),
            )
        ],
    )
    merged = merge_agent_claims(ledger, [batch], [weak])
    apply_claim_alignment([weak], [batch], [facet])
    row = compute_facet_coverage([facet], merged)[0]
    assert weak.facet_scores["guideline_concordance"] == text_score
    assert row.coverage_probability < facet.threshold
    assert row.status != "satisfied"
    assert all(claim.confidence <= text_score + 1e-9 for claim in merged[0].claims)


def test_a_recommendation_heading_does_not_clear_a_one_keyword_sentence():
    facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["apixaban"],
        required=True,
        threshold=0.70,
    )
    weak = _passage("weak", "See apixaban in the appendix.")
    weak.section = "Recommendations"
    weak.section_score = 1.0
    weak.calibrated_score = 0.95
    weak.evidence_grade_score = 0.95
    score_passage_facets(weak, [facet])
    ledger = build_retrieval_ledger([weak], [facet])
    batch = AgentClaimBatch(
        agent_id="guideline",
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is indicated.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="weak",
                confidence=1.0,
                calibrated_relevance=1.0,
                evidence_grade=EvidenceGrade(score=1.0),
            )
        ],
    )
    merged = merge_agent_claims(ledger, [batch], [weak])
    row = compute_facet_coverage([facet], merged)[0]
    assert weak.facet_scores["guideline_concordance"] < 0.50
    assert row.coverage_probability < facet.threshold
    assert row.status != "satisfied"


def test_a_specialist_confidence_cannot_cover_a_facet_the_passage_does_not_support():
    passage = _passage("cpg-1", "Metformin is first-line for type 2 diabetes.")
    passage.calibrated_score = 1.0
    passage.evidence_grade_score = 1.0
    facets = [
        ClinicalFacet(name="guideline_concordance", keywords=["metformin", "first-line"], threshold=0.70),
        ClinicalFacet(name="safety_contraindications", keywords=["contraindicated"], threshold=0.70),
    ]
    score_passage_facets(passage, facets)
    ledger = build_retrieval_ledger([passage], facets)
    batch = AgentClaimBatch(
        agent_id="safety",
        claims=[
            EvidenceClaim(
                facet="safety_contraindications",
                claim="Metformin is contraindicated.",
                chunk_id="cpg-1",
                confidence=1.0,
                calibrated_relevance=1.0,
                evidence_grade=EvidenceGrade(score=1.0),
            ),
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Metformin is first-line.",
                chunk_id="cpg-1",
                confidence=0.9,
            ),
        ],
    )
    merged = merge_agent_claims(ledger, [batch], [passage])
    apply_claim_alignment([passage], [batch], facets)
    entry = merged[0]
    assert "safety_contraindications" not in entry.facets
    assert "guideline_concordance" in entry.facets
    assert passage.facet_scores["safety_contraindications"] == 0.0
    coverage = {row.facet: row.status for row in compute_facet_coverage(facets, merged)}
    assert coverage["safety_contraindications"] == "missing"
    assert coverage["guideline_concordance"] == "satisfied"


def test_a_specialist_cannot_turn_a_caution_into_support():
    passage = _passage("cpg-1", "Avoid metformin when the eGFR is low.")
    passage.calibrated_score = 0.95
    passage.evidence_grade_score = 0.95
    facet = ClinicalFacet(
        name="guideline_concordance",
        keywords=["metformin"],
        required=True,
        threshold=0.70,
    )
    score_passage_facets(passage, [facet])
    ledger = build_retrieval_ledger([passage], [facet])
    batch = AgentClaimBatch(
        agent_id="guideline",
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Metformin is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="cpg-1",
                confidence=1.0,
                calibrated_relevance=1.0,
                evidence_grade=EvidenceGrade(score=1.0),
            )
        ],
    )
    merged = merge_agent_claims(ledger, [batch], [passage])
    coverage = compute_facet_coverage([facet], merged)
    assert coverage[0].coverage_probability == 0.0
    assert coverage[0].supporting_chunk_ids == []
    assert all(claim.polarity != EvidencePolarity.SUPPORTS for claim in merged[0].claims)


def test_a_caution_confidence_does_not_outrank_the_recommendation():
    from app.scoring.utility import select_bundle

    facet = ClinicalFacet(name="guideline_concordance", keywords=["apixaban"], required=True, threshold=0.70)
    recommendation = _passage("rec", "Apixaban is recommended for atrial fibrillation.")
    caution = _passage("caut", "Avoid apixaban. Apixaban is contraindicated.", source_type=SourceType.SAFETY)
    for passage in (recommendation, caution):
        passage.calibrated_score = 0.90
        passage.fusion_score = 0.80
        passage.token_count = 40
    caution.contradiction_score = 0.32
    score_passage_facets(recommendation, [facet])
    score_passage_facets(caution, [facet])
    batch = AgentClaimBatch(
        agent_id="guideline",
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="caut",
                confidence=1.0,
            )
        ],
    )
    apply_claim_alignment([recommendation, caution], [batch], [facet])
    assert caution.facet_scores["guideline_concordance"] < 1.0
    selected = select_bundle([recommendation, caution], token_budget=40, max_chunks=1)
    assert [item.chunk_id for item in selected] == ["rec"]


def test_a_chart_caution_conflicts_with_the_same_drug():
    support = _entry(
        "rec",
        "guideline_concordance",
        "Apixaban is recommended.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    support.text = "Apixaban is recommended."
    pairs = chart_contradictions("Avoid apixaban.", [support])
    assert len(pairs) == 1
    assert pairs[0].chunk_id_b == PATIENT_CHART_ID
    assert pairs[0].severity == "high"
    assert "apixaban" in pairs[0].reason
    assert chart_contradictions("Avoid dabigatran.", [support]) == []
    assert chart_contradictions("ASA 100mg daily", [support]) == []
    assert chart_contradictions("The ASA meeting used a 100mg slide.", [support]) == []
    metformin = _entry(
        "met",
        "guideline_concordance",
        "Metformin is recommended.",
        EvidencePolarity.SUPPORTS,
        grade=0.90,
    )
    metformin.text = "Metformin is recommended."
    assert chart_contradictions("Avoid apixaban. Metformin is continued.", [metformin]) == []
    mixed = chart_contradictions("Avoid apixaban. Metformin is continued.", [support])
    assert mixed
    assert mixed[0].reason.startswith("Chart caution: Avoid apixaban")
    assert "Metformin is continued" not in mixed[0].reason
    from app.schemas.enums import ClinicalScope

    gate = EvidenceGate()
    facet = ClinicalFacet(name="guideline_concordance", required=False, threshold=0.1)
    decision = gate.decide(
        [support],
        [facet],
        [],
        2,
        ClinicalScope.CLINICIAN_CDS,
        contradictions=pairs,
        stop_retrieval=True,
    )
    assert decision.passed is False
    assert "apixaban" in decision.reason


def test_drug_tokens_fold_plurals_and_ignore_april():
    assert intervention_tokens("Avoid NSAIDs") == {"nsaid"}
    assert same_intervention("Avoid NSAIDs", "NSAID is recommended") is True
    assert "april" not in intervention_tokens("Avoid therapy in April")
    assert same_intervention("Recommend lisinopril in April", "Avoid therapy in April") is False
    assert intervention_tokens("low-molecular-weight-heparin") == {"heparin"}
