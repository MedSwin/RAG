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
