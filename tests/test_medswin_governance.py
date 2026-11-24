from app.models.medswin import CandidatePassage, SourceType
from app.services.medswin.governance import build_citation, ensure_cds_language, redact_phi_payload, redact_phi_text


def test_a_long_passage_shows_the_chart_dose_first():
    from app.services.medswin.governance import lead_with_caution

    filler = " ".join(f"history{i}" for i in range(40))
    shown = lead_with_caution(f"{filler} ASA 100mg daily")
    assert shown.lower().startswith("asa 100mg daily")
    assert "history0" in shown
    assert shown.lower().find("asa 100mg") < shown.lower().find("history0")


def test_a_long_passage_shows_the_caution_first():
    from app.schemas.evidence import EvidenceBundle
    from app.schemas.traces import ChatResponse
    from app.services.medswin.governance import redact_public_chat

    history = " ".join(f"history{i}." for i in range(40))
    response = ChatResponse(
        answer="The evidence is insufficient.",
        evidence_bundle=EvidenceBundle(
            passages=[
                CandidatePassage(
                    chunk_id="chart-1",
                    doc_id="doc-chart",
                    source_type=SourceType.EMR,
                    text=f"{history} Avoid apixaban.",
                )
            ],
            total_tokens=20,
            cpg_count=0,
            emr_count=1,
            lit_count=0,
            safety_count=0,
        ),
        trace_id="trace-caution",
    )
    public = redact_public_chat(response)
    shown = public.evidence_bundle.passages[0].text
    assert shown.startswith("Avoid apixaban.")
    assert "apixaban" in shown[:320].lower()
    assert "history39" in shown


def test_a_long_sentence_shows_the_later_drug_first():
    from app.services.medswin.governance import lead_with_caution

    filler = " ".join(f"history{i}" for i in range(30))
    gap = (
        "this treatment immediately whenever the haematology service confirms "
        "persistent gastrointestinal bleeding after a recent major procedure involving"
    )
    shown = lead_with_caution(f"{filler} Avoid {gap} apixaban")
    assert "apixaban" in shown[:80].lower()
    assert shown.lower().find("apixaban") < shown.lower().find("history0")
    assert lead_with_caution("Avoid apixaban.").lower().count("avoid apixaban") == 1


def test_a_later_sentence_shows_the_caution_drug_first():
    from app.services.medswin.governance import lead_with_caution

    gap = (
        "this treatment immediately whenever the haematology service confirms "
        "persistent gastrointestinal bleeding after a recent major procedure involving"
    )
    caution = f"Avoid {gap} apixaban."
    history = " ".join(f"history{i}." for i in range(5))
    for text in (f"{history} {caution}", f"{caution} {history}"):
        shown = lead_with_caution(text)
        assert "apixaban" in shown[:80].lower()
        assert shown.lower().find("apixaban") < shown.lower().find("history0")


def test_the_chat_response_drops_a_phone_number():
    from app.schemas.evidence import EvidenceBundle, EvidenceClaim, EvidenceLedgerEntry
    from app.schemas.traces import ChatResponse
    from app.services.medswin.governance import redact_public_chat

    response = ChatResponse(
        answer="Call 555-234-5678. Continue metformin.",
        evidence_bundle=EvidenceBundle(
            passages=[
                CandidatePassage(
                    chunk_id="emr-1",
                    doc_id="doc-emr-1",
                    source_type=SourceType.EMR,
                    text="Call 555-234-5678. Creatinine is 1.4. Continue metformin.",
                    metadata={"patient_id": "pt-1", "note": "Call 555-234-5678."},
                )
            ],
            total_tokens=12,
            cpg_count=0,
            emr_count=1,
            lit_count=0,
            safety_count=0,
            evidence_ledger=[
                EvidenceLedgerEntry(
                    chunk_id="emr-1",
                    doc_id="doc-emr-1",
                    source_type=SourceType.EMR,
                    text="Call 555-234-5678. Creatinine is 1.4.",
                    claims=[
                        EvidenceClaim(
                            facet="patient_applicability",
                            claim="Call 555-234-5678. Creatinine is 1.4.",
                            chunk_id="emr-1",
                        )
                    ],
                )
            ],
        ),
        trace_id="trace-1",
    )
    public = redact_public_chat(response)
    assert "555-234-5678" not in public.answer
    assert "metformin" in public.answer
    passage = public.evidence_bundle.passages[0]
    assert "555-234-5678" not in passage.text
    assert "Creatinine is 1.4." in passage.text
    assert passage.metadata["patient_id"] == "pt-1"
    assert "555-234-5678" not in passage.metadata["note"]
    assert "555-234-5678" not in public.evidence_bundle.evidence_ledger[0].text
    assert "Creatinine is 1.4." in public.evidence_bundle.evidence_ledger[0].claims[0].claim
    assert public.evidence_bundle.passages[0].chunk_id == "emr-1"


def test_phi_redaction_handles_direct_identifiers():
    text = "Patient ID: ABC1234, DOB: 01/02/1970, call +1 555 123 4567 or test@example.com"

    redacted = redact_phi_text(text)

    assert "ABC1234" not in redacted
    assert "test@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_the_retrieval_marker_is_not_redacted_as_a_patient_id():
    from app.services.medswin.governance import redact_phi_text

    text = "Which metformin dose?\n\nPatient retrieval context:\nAvoid apixaban."
    redacted = redact_phi_text(text)
    assert "Patient retrieval context:" in redacted
    assert "Avoid apixaban." in redacted
    assert "metformin" in redacted
    assert "[REDACTED_PATIENT_ID]" not in redacted
    labelled = redact_phi_text("Review MRN: ABCD99 and Patient MRN: 123456.")
    assert "ABCD99" not in labelled
    assert "123456" not in labelled
    assert labelled.count("[REDACTED_PATIENT_ID]") == 2


def test_recursive_phi_redaction_and_cds_boundary():
    payload = {"query": "MRN: XYZZY99 needs review", "items": ["email a@b.com"]}

    redacted = redact_phi_payload(payload)
    answer = ensure_cds_language("Evidence supports clinician review.")

    assert "XYZZY99" not in redacted["query"]
    assert "a@b.com" not in redacted["items"][0]
    assert "does not establish a final diagnosis" in answer


def test_a_stored_trace_redacts_free_text_and_keeps_ids(monkeypatch):
    import asyncio

    from app.repositories.traces import TraceRepository
    from app.schemas.traces import AuditTrace

    class _Store:
        def __init__(self):
            self.saved = None

        async def insert_one(self, data):
            self.saved = data

            class _Result:
                inserted_id = "row-1"

            return _Result()

    monkeypatch.setattr("app.repositories.traces.settings.LOG_REDACT_PHI", True)
    store = _Store()
    repo = TraceRepository()
    repo._db = { "traces": store }
    trace = AuditTrace(
        trace_id="trace-1",
        session_id="sess-1",
        user_id="dr-chen",
        org_id="org-a",
        patient_id="pt-1",
        hook_instance="555-234-5678",
        generator_profile="medswin",
        query="Call the ward at 555-234-5678 about MRN: ABCD99",
    )
    asyncio.run(repo.create(trace, "org-a"))
    assert store.saved["trace_id"] == "trace-1"
    assert store.saved["patient_id"] == "pt-1"
    assert store.saved["hook_instance"] == "555-234-5678"
    assert store.saved["generator_profile"] == "medswin"
    assert "555-234-5678" not in store.saved["query"]
    assert "ABCD99" not in store.saved["query"]
    assert "[REDACTED_PHONE]" in store.saved["query"]
    monkeypatch.setattr("app.repositories.traces.settings.LOG_REDACT_PHI", False)
    open_store = _Store()
    repo._db = {"traces": open_store}
    asyncio.run(repo.create(trace, "org-a"))
    assert "555-234-5678" in open_store.saved["query"]


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
