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


def test_the_retrieval_response_shows_a_late_caution():
    from app.api.v1.endpoints.retrieval import _to_response

    filler = "History note. " * 40
    passage = _candidate("emr-late", f"{filler} Avoid apixaban.", source_type=SourceType.EMR)
    document = _to_response(passage)
    assert document.content.startswith("Avoid apixaban.")
    assert "History note." in document.content


def test_the_retrieval_response_drops_a_phone_number():
    from app.api.v1.endpoints.retrieval import _public_text, _to_response

    passage = _candidate(
        "emr-1",
        "Call 555-234-5678. Creatinine is 1.4. Continue metformin.",
        source_type=SourceType.EMR,
    )
    passage.metadata = {
        "patient_id": "pt-1",
        "note": "Call 555-234-5678 about the creatinine result.",
        "question": "Call 555-234-5678",
        "context": "Creatinine is 1.4.",
        "answer": "Continue metformin.",
    }
    document = _to_response(passage)
    assert "555-234-5678" not in document.content
    assert "[REDACTED_PHONE]" in document.content
    assert "Creatinine is 1.4." in document.content
    assert "metformin" in document.content
    assert document.metadata["patient_id"] == "pt-1"
    assert "555-234-5678" not in document.metadata["note"]
    assert document.question == "Call [REDACTED_PHONE]"
    assert document.context == "Creatinine is 1.4."
    assert _public_text("dose for 555-234-5678") == "dose for [REDACTED_PHONE]"


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


def test_the_reranker_log_keeps_the_query_phone_out(caplog):
    import logging

    from app.services.reranker import DocumentReranker

    reranker = DocumentReranker.__new__(DocumentReranker)

    def score(_query, doc):
        raise RuntimeError(doc)

    reranker.compute_relevance_score = score
    with caplog.at_level(logging.INFO):
        ranked = reranker.rerank_documents(
            "dose for 555-234-5678",
            [{"chunk_id": "a", "text": "secret chart text"}],
        )
    assert ranked == []
    assert "555-234-5678" not in caplog.text
    assert "secret chart text" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_the_local_reranker_drops_a_phone_in_the_passage():
    import asyncio

    from app.services.adapters.reranker import RerankerClient
    from app.services.reranker import DocumentReranker

    seen = {}

    def score(query, doc):
        seen["query"] = query
        seen["doc"] = doc
        return 0.4

    reranker = DocumentReranker.__new__(DocumentReranker)
    reranker.compute_relevance_score = score
    reranker.rerank_documents(
        "dose for 555-234-5678",
        [{"chunk_id": "a", "text": "Call 555-234-5678. Creatinine is 1.4. Continue metformin."}],
    )
    assert "555-234-5678" not in seen["query"]
    assert "555-234-5678" not in seen["doc"]
    assert "[REDACTED_PHONE]" in seen["doc"]
    assert "Creatinine is 1.4." in seen["doc"]
    assert "metformin" in seen["doc"]

    client = RerankerClient.__new__(RerankerClient)
    client.provider = "local"

    async def local(query, passages):
        seen["client_query"] = query
        seen["client_passages"] = passages
        return [{"index": 0, "score": 0.5}]

    client._wants_inprocess_local = lambda: True
    client._local_rerank = local
    asyncio.run(
        client.rerank(
            "dose for 555-234-5678",
            ["Call 555-234-5678. Creatinine is 1.4. Continue metformin."],
        )
    )
    assert "555-234-5678" not in seen["client_query"]
    assert "555-234-5678" not in seen["client_passages"][0]
    assert "Creatinine is 1.4." in seen["client_passages"][0]


def test_the_local_reranker_scores_a_late_caution():
    import torch
    from app.services.reranker import DocumentReranker

    reranker = DocumentReranker.__new__(DocumentReranker)
    reranker.model_type = "bge-m3"
    reranker.device = torch.device("cpu")
    seen = {}

    class _Tokenizer:
        def __call__(self, queries, documents, truncation, padding, max_length, return_tensors):
            seen["documents"] = list(documents)
            ids = torch.ones((len(documents), 2), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    class _Model:
        def __call__(self, **inputs):
            class _Out:
                logits = torch.zeros((inputs["input_ids"].shape[0], 1))

            return _Out()

    reranker.tokenizer = _Tokenizer()
    reranker.model = _Model()
    filler = "History note. " * 80
    reranker.compute_relevance_scores("Which dose?", [f"{filler} Avoid apixaban."])
    assert seen["documents"][0].startswith("Avoid apixaban.")
    assert "History note." in seen["documents"][0]


def test_document_reranker_scores_text_and_content_chunks():
    from app.services.reranker import DocumentReranker

    reranker = DocumentReranker.__new__(DocumentReranker)
    reranker.compute_relevance_score = lambda query, doc: 0.9 if "aspirin" in doc.lower() else 0.1
    ranked = reranker.rerank_documents(
        "aspirin",
        [
            {"chunk_id": "b", "content": "Unrelated astronomy sentence."},
            {"chunk_id": "a", "text": "Aspirin inhibits platelet aggregation."},
        ],
        top_k=2,
    )
    assert [row["chunk_id"] for row in ranked] == ["a", "b"]
    assert ranked[0]["rerank_score"] == 0.9


def test_a_sentence_period_does_not_hide_a_chart_drug():
    from app.retrieval.lexical import BM25Okapi, LexicalRetriever, _matching_indexes

    query = LexicalRetriever._bm25_tokens("Avoid apixaban.")
    passage = LexicalRetriever._bm25_tokens("Avoid apixaban when bleeding is present.")
    assert "apixaban" in query
    assert "apixaban" in passage
    index = BM25Okapi([passage, ["unrelated"]])
    assert _matching_indexes(index, query, 2) == [0]


def test_a_later_chart_drug_stays_in_the_search_terms():
    from app.retrieval.lexical import LexicalRetriever

    drugs = [
        "apixaban",
        "warfarin",
        "heparin",
        "enoxaparin",
        "metformin",
        "aspirin",
        "dabigatran",
        "rivaroxaban",
    ]
    sentences = [
        f"Avoid alpha{index}a alpha{index}b alpha{index}c {drug}."
        for index, drug in enumerate(drugs)
    ]
    query = "Which dose?\n\nPatient retrieval context:\n" + " ".join(sentences)
    terms = LexicalRetriever._lexical_terms(query)
    for drug in drugs:
        assert drug in terms


def test_a_chart_dose_at_the_end_of_a_long_note_stays_in_the_search():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = " ".join(f"history{i}" for i in range(30)) + " ASA 100mg daily"
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "asa" in terms
    assert "aspirin" in terms
    assert "history0" not in terms
    assert "history29" not in terms
    assert "aspirin" in focus.lower()
    assert "history0" not in focus.lower()
    meeting = "The ASA meeting used a 100mg slide " + " ".join(f"history{i}" for i in range(12))
    meeting_terms = LexicalRetriever._lexical_terms(
        f"Which topic?\n\nPatient retrieval context:\n{_chart_synopsis([{'text': meeting}])}"
    )
    assert "aspirin" not in meeting_terms


def test_an_aspirin_abbreviation_also_searches_aspirin():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid ASA " + " ".join(f"history{i}" for i in range(20))
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "asa" in terms
    assert "aspirin" in terms
    assert "history0" not in terms
    assert "aspirin" in focus.lower()
    assert "history0" not in focus.lower()
    meeting = LexicalRetriever._lexical_terms("What did the ASA meeting decide?")
    assert "aspirin" not in meeting
    dose = LexicalRetriever._lexical_terms("Which ASA dose?")
    focus = LexicalRetriever.focus_query("Which ASA dose?")
    assert "asa" in dose
    assert "aspirin" in dose
    assert "aspirin" in focus.lower()
    glued = LexicalRetriever._lexical_terms("ASA 100mg")
    glued_focus = LexicalRetriever.focus_query("ASA 100mg")
    assert "asa" in glued
    assert "aspirin" in glued
    assert "aspirin" in glued_focus.lower()
    meeting_dose = LexicalRetriever._lexical_terms("The ASA meeting used a 100mg slide")
    assert "aspirin" not in meeting_dose


def test_a_brand_caution_also_searches_the_generic_drug():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid Eliquis " + " ".join(f"history{i}" for i in range(20))
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "eliquis" in terms
    assert "apixaban" in terms
    assert "history0" not in terms
    assert "eliquis" in focus.lower()
    assert "apixaban" in focus.lower()
    assert "history0" not in focus.lower()
    question_terms = LexicalRetriever._lexical_terms("Which Eliquis dose?")
    assert "eliquis" in question_terms
    assert "apixaban" in question_terms


def test_a_chart_drug_after_leading_history_still_blocks_the_same_drug():
    from app.medswin.ledger import chart_contradictions
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever
    from app.schemas.enums import EvidencePolarity, SourceType
    from app.schemas.evidence import EvidenceClaim, EvidenceGrade, EvidenceLedgerEntry

    filler = " ".join(f"history{i}" for i in range(150))
    note = f"{filler} Avoid this oral therapy after the bleeding risk review for apixaban"
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    assert "apixaban" in synopsis.lower()
    assert "apixaban" in terms
    assert "history0" not in terms
    assert "history149" not in terms
    support = EvidenceLedgerEntry(
        chunk_id="rec",
        doc_id="doc-rec",
        source_type=SourceType.CPG,
        text="Apixaban is recommended.",
        evidence_grade=EvidenceGrade(score=0.90),
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="rec",
                confidence=0.9,
            )
        ],
    )
    pairs = chart_contradictions(synopsis, [support])
    assert pairs
    assert "apixaban" in pairs[0].reason


def test_a_drug_past_twelve_words_stays_in_the_same_caution_sentence():
    from app.medswin.ledger import chart_contradictions
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever
    from app.schemas.enums import EvidencePolarity, SourceType
    from app.schemas.evidence import EvidenceClaim, EvidenceGrade, EvidenceLedgerEntry

    gap = (
        "this treatment immediately whenever the haematology service confirms "
        "persistent gastrointestinal bleeding after a recent major procedure involving"
    )
    assert len(gap.split()) > 12
    note = (
        f"Avoid {gap} apixaban. Metformin is continued. "
        + " ".join(f"history{i}" for i in range(20))
    )
    synopsis = _chart_synopsis([{"text": note}])
    terms = LexicalRetriever._lexical_terms(
        f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    )
    assert "apixaban" in terms
    assert "history0" not in terms
    support = EvidenceLedgerEntry(
        chunk_id="rec",
        doc_id="doc-rec",
        source_type=SourceType.CPG,
        text="Apixaban is recommended.",
        evidence_grade=EvidenceGrade(score=0.90),
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="rec",
                confidence=0.9,
            )
        ],
    )
    pairs = chart_contradictions(synopsis, [support])
    assert pairs
    assert "apixaban" in pairs[0].reason
    assert "metformin" not in pairs[0].reason.lower()


def test_a_second_drug_in_the_same_caution_sentence_still_blocks():
    from app.medswin.ledger import chart_contradictions
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever
    from app.schemas.enums import EvidencePolarity, SourceType
    from app.schemas.evidence import EvidenceClaim, EvidenceGrade, EvidenceLedgerEntry

    gap = (
        "after the prolonged bleeding risk that applies during a recent admission "
        "and the later clinic review also names"
    )
    assert len(gap.split()) > 4
    note = (
        f"Avoid warfarin {gap} apixaban. Metformin is continued. "
        + " ".join(f"history{i}" for i in range(20))
    )
    synopsis = _chart_synopsis([{"text": note}])
    terms = LexicalRetriever._lexical_terms(
        f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    )
    assert "warfarin" in terms
    assert "apixaban" in terms
    assert "history0" not in terms
    assert "metformin" not in terms

    def support(chunk_id: str, text: str) -> EvidenceLedgerEntry:
        return EvidenceLedgerEntry(
            chunk_id=chunk_id,
            doc_id=chunk_id,
            source_type=SourceType.CPG,
            text=text,
            evidence_grade=EvidenceGrade(score=0.90),
            claims=[
                EvidenceClaim(
                    facet="guideline_concordance",
                    claim=text,
                    polarity=EvidencePolarity.SUPPORTS,
                    chunk_id=chunk_id,
                    confidence=0.9,
                )
            ],
        )

    apixaban_pairs = chart_contradictions(synopsis, [support("apix", "Apixaban is recommended.")])
    warfarin_pairs = chart_contradictions(synopsis, [support("warf", "Warfarin is recommended.")])
    metformin_pairs = chart_contradictions(synopsis, [support("met", "Metformin is recommended.")])
    assert apixaban_pairs and "apixaban" in apixaban_pairs[0].reason
    assert warfarin_pairs and "warfarin" in warfarin_pairs[0].reason
    assert metformin_pairs == []


def test_a_drug_before_the_caution_word_still_blocks():
    from app.medswin.ledger import chart_contradictions
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever
    from app.schemas.enums import EvidencePolarity, SourceType
    from app.schemas.evidence import EvidenceClaim, EvidenceGrade, EvidenceLedgerEntry

    lead = (
        "Apixaban remains the selected agent after the prolonged admission "
        "and the bleeding review says"
    )
    assert len(lead.split()) > 4
    note = (
        f"{lead} avoid therapy in active bleeding. Metformin is continued. "
        + " ".join(f"history{i}" for i in range(20))
    )
    synopsis = _chart_synopsis([{"text": note}])
    terms = LexicalRetriever._lexical_terms(
        f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    )
    assert "apixaban" in terms
    assert "metformin" not in terms
    assert "history0" not in terms
    support = EvidenceLedgerEntry(
        chunk_id="rec",
        doc_id="rec",
        source_type=SourceType.CPG,
        text="Apixaban is recommended.",
        evidence_grade=EvidenceGrade(score=0.90),
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Apixaban is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="rec",
                confidence=0.9,
            )
        ],
    )
    pairs = chart_contradictions(synopsis, [support])
    assert pairs
    assert "apixaban" in pairs[0].reason
    other = EvidenceLedgerEntry(
        chunk_id="met",
        doc_id="met",
        source_type=SourceType.CPG,
        text="Metformin is recommended.",
        evidence_grade=EvidenceGrade(score=0.90),
        claims=[
            EvidenceClaim(
                facet="guideline_concordance",
                claim="Metformin is recommended.",
                polarity=EvidencePolarity.SUPPORTS,
                chunk_id="met",
                confidence=0.9,
            )
        ],
    )
    assert chart_contradictions(synopsis, [other]) == []


def test_a_drug_several_words_after_the_caution_stays_in_the_search():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid this oral tablet form of apixaban " + " ".join(f"history{i}" for i in range(20))
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "apixaban" in terms
    assert "history0" not in terms
    assert "history19" not in terms
    assert "apixaban" in focus
    assert "history0" not in focus


def test_history_between_two_caution_drugs_stays_out_of_the_search():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid warfarin " + " ".join(f"history{i}" for i in range(12)) + " avoid apixaban"
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "warfarin" in terms
    assert "apixaban" in terms
    assert "history0" not in terms
    assert "history11" not in terms
    assert "warfarin" in focus
    assert "apixaban" in focus
    assert "history0" not in focus


def test_a_second_drug_in_the_same_caution_stays_in_the_search():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid warfarin and also avoid apixaban " + " ".join(f"history{i}" for i in range(20))
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "warfarin" in terms
    assert "apixaban" in terms
    assert "history19" not in terms
    assert "history0" not in terms
    assert "apixaban" in focus
    assert "history0" not in focus


def test_a_caution_without_a_period_drops_the_following_history():
    from app.medswin.orchestrator import _chart_synopsis
    from app.retrieval.lexical import LexicalRetriever

    note = "Avoid apixaban " + " ".join(f"history{i}" for i in range(30))
    synopsis = _chart_synopsis([{"text": note}])
    query = f"Which metformin dose?\n\nPatient retrieval context:\n{synopsis}"
    terms = LexicalRetriever._lexical_terms(query)
    focus = LexicalRetriever.focus_query(query)
    assert "apixaban" in terms
    assert "metformin" in terms
    assert "history29" not in terms
    assert "history0" not in terms
    assert "history29" not in focus
    assert "history0" not in focus


def test_a_question_word_does_not_outrank_the_question_drug():
    from app.retrieval.lexical import BM25Okapi, LexicalRetriever, _matching_indexes

    terms = LexicalRetriever._lexical_terms("Which metformin dose for this patient?")
    assert "which" not in terms
    assert "for" not in terms
    assert "this" not in terms
    assert "metformin" in terms
    assert "dose" in terms
    filler = LexicalRetriever._bm25_tokens("Which form is this for.")
    drug = LexicalRetriever._bm25_tokens("Metformin is first-line treatment.")
    docs = [filler, drug]
    docs += [LexicalRetriever._bm25_tokens(f"Unrelated topic number {i} about diet.") for i in range(8)]
    order = _matching_indexes(BM25Okapi(docs), terms, 3)
    assert 0 not in order
    assert order[0] == 1


def test_a_long_chart_keeps_the_question_drug_in_the_lexical_query():
    from app.retrieval.lexical import BM25Okapi, LexicalRetriever, _matching_indexes

    context = "Avoid apixaban. " + " ".join(f"history{i}" for i in range(80))
    query = f"Which metformin dose?\n\nPatient retrieval context:\n{context}"
    terms = LexicalRetriever._fts_query(query).split(" OR ")
    assert '"avoid"' in terms[:8]
    assert '"apixaban"' in terms[:8]
    assert '"metformin"' in terms
    assert '"history79"' not in terms
    assert len(terms) <= 64
    tokens = LexicalRetriever._lexical_terms(query)
    assert tokens == [term.strip('"') for term in terms]
    history = LexicalRetriever._bm25_tokens(" ".join(f"history{i} note" for i in range(40)))
    drug = LexicalRetriever._bm25_tokens("Metformin is first-line treatment.")
    caution = LexicalRetriever._bm25_tokens("Avoid apixaban when bleeding is present.")
    order = _matching_indexes(BM25Okapi([history, drug, caution]), tokens, 3)
    assert 0 not in order
    assert order[0] == 2


def test_a_long_question_keeps_the_chart_caution_in_the_lexical_query():
    from app.retrieval.lexical import LexicalRetriever

    question = " ".join(["doseword"] * 80)
    query = f"{question}\n\nPatient retrieval context:\nAvoid apixaban."
    match = LexicalRetriever._fts_query(query)
    assert '"avoid"' in match.split(" OR ")[:8]
    assert '"apixaban"' in match.split(" OR ")[:8]
    assert len(match.split(" OR ")) <= 64


def test_a_lexical_caution_stays_when_dense_hits_fill_the_pool():
    from app.retrieval.hybrid import _with_lexical_cautions

    fillers = [
        _candidate(f"dense-{index}", "A general note.", dense=0.90)
        for index in range(80)
    ]
    for item in fillers:
        item.lexical_score = None
    other = _candidate("other", "A trial report about methods.", dense=None)
    other.lexical_score = 5.0
    other.dense_score = None
    caution = _candidate("caution", "Avoid apixaban.", source_type=SourceType.SAFETY, dense=None)
    caution.lexical_score = 1.0
    caution.dense_score = None
    ordered = sorted(
        [*fillers, other, caution],
        key=lambda item: (item.dense_score or 0.0, item.lexical_score or 0.0),
        reverse=True,
    )
    pool = _with_lexical_cautions(ordered, 80)
    kept = {item.chunk_id for item in pool}
    assert "caution" in kept
    assert "other" not in kept
    assert len(pool) == 81


def test_a_lexical_caution_stays_ahead_of_weaker_dense_cautions():
    from app.retrieval.hybrid import _with_lexical_cautions

    fillers = [
        _candidate(f"dense-{index}", "A general note.", dense=0.90)
        for index in range(80)
    ]
    for item in fillers:
        item.lexical_score = None
    dense_cautions = []
    for index in range(12):
        item = _candidate(
            f"weak-{index}",
            "Avoid warfarin.",
            source_type=SourceType.SAFETY,
            dense=0.30 - (index * 0.01),
        )
        item.lexical_score = None
        dense_cautions.append(item)
    lexical = _candidate("lexical", "Avoid apixaban.", source_type=SourceType.SAFETY, dense=None)
    lexical.lexical_score = 1.0
    lexical.dense_score = None
    ordered = sorted(
        [*fillers, *dense_cautions, lexical],
        key=lambda item: (item.dense_score or 0.0, item.lexical_score or 0.0),
        reverse=True,
    )
    pool = _with_lexical_cautions(ordered, 80)
    kept = {item.chunk_id for item in pool}
    assert "lexical" in kept
    assert "weak-0" in kept
    assert "weak-11" not in kept


def test_the_retrieval_trace_lists_a_caution_past_the_first_rows():
    from app.retrieval.hybrid import _trace_candidates

    fillers = [
        _candidate(f"dense-{index}", "A general note.", dense=0.90)
        for index in range(40)
    ]
    caution = _candidate("caution", "Avoid apixaban.", source_type=SourceType.SAFETY, dense=0.10)
    other = _candidate("other", "A trial report about methods.", dense=0.05)
    rows = _trace_candidates([*fillers, caution, other], limit=40)
    ids = [row["chunk_id"] for row in rows]
    assert ids.count("caution") == 1
    assert "other" not in ids
