from app.medswin.abstain import abstention_matches, citations_match
from app.medswin.gate import EvidenceGate
from app.medswin.ledger import (
    adjudicate_contradictions,
    apply_claim_alignment,
    build_retrieval_ledger,
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
    assert "apixaban" not in ledger[0].claims[0].claim.lower()
    pairs = detect_contradictions(ledger)
    assert any(pair.chunk_id_a == "cpg-late" and pair.chunk_id_b == "safety-late" for pair in pairs)


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


def test_monitoring_does_not_qualify_the_same_drug():
    facet = ClinicalFacet(name="guideline_concordance", keywords=["metformin"])
    support = _passage("cpg", "Metformin is recommended for type 2 diabetes.")
    monitor = _passage("lab", "Laboratory monitoring is recommended during metformin treatment.")
    caution = _passage("ckd", "Use metformin with caution in CKD.")
    support.calibrated_score = 0.9
    monitor.calibrated_score = 0.9
    ledger = build_retrieval_ledger([support, monitor], [facet])
    assert infer_polarity(monitor, facet) == EvidencePolarity.SUPPORTS
    assert infer_polarity(caution, facet) == EvidencePolarity.QUALIFIES
    assert detect_contradictions(ledger) == []


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


def test_a_trace_update_keeps_the_owner_and_the_identity_fields():
    import asyncio

    from app.repositories.traces import TraceRepository

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
    assert store.calls[1][1]["user_id"] == "dr-chen"
    assert len(store.calls) == 2


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
    assert scores["safety_contraindications"] == 0.0
    assert scores["guideline_concordance"] >= 0.95
    ledger = build_retrieval_ledger([passage], facets)
    assert "safety_contraindications" not in ledger[0].facets
    assert "guideline_concordance" in ledger[0].facets


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


def test_drug_tokens_fold_plurals_and_ignore_april():
    assert intervention_tokens("Avoid NSAIDs") == {"nsaid"}
    assert same_intervention("Avoid NSAIDs", "NSAID is recommended") is True
    assert "april" not in intervention_tokens("Avoid therapy in April")
    assert same_intervention("Recommend lisinopril in April", "Avoid therapy in April") is False
    assert intervention_tokens("low-molecular-weight-heparin") == {"heparin"}
