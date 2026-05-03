from app.api.v1.endpoints.medswin import _coerce_evidence_grade, _section_aware_chunks
from app.models.medswin import SourceType


def test_section_aware_chunks_preserve_headings_and_offsets():
    text = "Recommendations\n\nUse treatment when indicated.\n\nContraindications\n\nAvoid in severe allergy."

    chunks = _section_aware_chunks("doc1", text, None)

    assert len(chunks) == 2
    assert chunks[0]["section"] == "Recommendations"
    assert chunks[1]["section"] == "Contraindications"
    assert chunks[0]["offset_start"] is not None
    assert chunks[1]["offset_end"] == len(text)


def test_evidence_grade_defaults_follow_source_type():
    cpg_grade = _coerce_evidence_grade(None, SourceType.CPG)
    emr_grade = _coerce_evidence_grade(None, SourceType.EMR)

    assert cpg_grade.score > emr_grade.score
    assert emr_grade.label == "emr"


def test_section_aware_chunks_handles_empty_text():
    assert _section_aware_chunks("doc1", "", None) == []
