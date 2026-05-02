from app.models.medswin import CandidatePassage, SourceType
from app.services.medswin.governance import build_citation, ensure_cds_language, redact_phi_payload, redact_phi_text


def test_phi_redaction_handles_direct_identifiers():
    text = "Patient ID: ABC1234, DOB: 01/02/1970, call +1 555 123 4567 or test@example.com"

    redacted = redact_phi_text(text)

    assert "ABC1234" not in redacted
    assert "test@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_recursive_phi_redaction_and_cds_boundary():
    payload = {"query": "MRN: XYZZY99 needs review", "items": ["email a@b.com"]}

    redacted = redact_phi_payload(payload)
    answer = ensure_cds_language("Evidence supports clinician review.")

    assert "XYZZY99" not in redacted["query"]
    assert "a@b.com" not in redacted["items"][0]
    assert "does not establish a final diagnosis" in answer


def test_citation_contains_provenance_and_policy_metadata():
    passage = CandidatePassage(
        chunk_id="c1",
        doc_id="d1",
        source_type=SourceType.CPG,
        text="Guideline recommendation.",
        section="Recommendations",
        offset_start=5,
        offset_end=42,
        rerank_score=0.8,
        fusion_score=0.9,
        metadata={"guideline_version": "2026.1", "effective_date": "2026-01-01"},
    )

    citation = build_citation(passage, facets=["guideline_concordance"])

    assert citation["guideline_version"] == "2026.1"
    assert citation["facets"] == ["guideline_concordance"]
    assert citation["evidence_grade"]["score"] >= 0.9
