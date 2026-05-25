import pytest

from app.core.config import Settings as RuntimeSettings
from eval.app.audit import audit_case, groundedness_proxy, trace_completeness
from eval.app.config import Settings
from eval.app.runner import _validate_qrel_coverage, _validate_setup_stats, run_benchmark
from eval.app.schemas import BenchmarkCase, GoldFacet, RunRequest


class _FakeClient:
    instances = []

    def __init__(
        self,
        base_url: str,
        org_id: str,
        user_id: str,
        timeout_s: float = 120.0,
        include_patient_context_in_query: bool = False,
        storage_stats_payload: dict | None = None,
    ):
        self.base_url = base_url
        self.org_id = org_id
        self.user_id = user_id
        self.timeout_s = timeout_s
        self.include_patient_context_in_query = include_patient_context_in_query
        self.storage_stats_payload = storage_stats_payload
        self.ingest_calls = []
        self.chat_queries = []
        self.trace_calls = []
        self.request_stats = {"retries": 0, "rate_limits": 0, "timeouts": 0, "network_errors": 0}
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def health(self):
        return {"status": "ok"}

    async def storage_stats(self, org_id=None):
        active_settings = RuntimeSettings()
        if self.storage_stats_payload is not None:
            return self.storage_stats_payload
        return {
            "source_counts": {"CPG": 0, "EMR": 0, "LIT": 1},
            "index_exists": True,
            "index_manifest": {
                "org_id": org_id,
                "embedding_space": active_settings.active_embedding_space(),
                "embedding_model": active_settings.active_embedding_model(),
                "embedding_dim": active_settings.active_embedding_dimension(),
                "total_vectors": 1,
            },
            "index_provenance_valid": True,
            "active_embedding_space": active_settings.active_embedding_space(),
            "active_embedding_model": active_settings.active_embedding_model(),
            "active_embedding_dim": active_settings.active_embedding_dimension(),
        }

    async def ingest_case_context(self, case):
        self.ingest_calls.append(case.case_id)
        return {"ingested": case.case_id}

    async def chat(self, case, *, source_policy, guideline_only, min_evidence_grade, clinical_scope):
        self.chat_queries.append(case.query)
        return {
            "trace_id": f"trace-{case.case_id}",
            "answer": "Clinical decision support only.",
            "policy_decision": {"passed": True},
            "citations": [{"doc_id": "doc-1"}],
            "evidence_bundle": {"passages": [], "evidence_ledger": []},
        }

    async def trace(self, trace_id, include_details=True):
        self.trace_calls.append((trace_id, include_details))
        return {
            "trace_id": trace_id,
            "messages_count": 1,
            "tool_calls_count": 1,
            "sufficiency_checks_count": 1,
            "evidence_passages_count": 1,
        }


@pytest.mark.asyncio
async def test_run_benchmark_uses_fixed_org_and_keeps_query_pure(monkeypatch, tmp_path):
    _FakeClient.instances = []
    case = BenchmarkCase(
        case_id="case-1",
        dataset="sample",
        query="What therapy is safest?",
        patient_id="patient-1",
        patient_context="Patient context that should be ingested, not pasted into the query.",
        gold_facets=[GoldFacet(facet_id="treatment", name="treatment", critical=True)],
    )

    monkeypatch.setattr("eval.app.runner.read_jsonl_cases", lambda path: [case])
    monkeypatch.setattr("eval.app.runner.MedSwinClient", _FakeClient)

    settings = Settings(
        medswin_base_url="http://medswin.test",
        benchmark_org_id="bench-org",
        benchmark_user_id="bench-user",
        request_timeout_s=5.0,
        run_store_dir=str(tmp_path),
        max_cases_default=25,
    )

    run = await run_benchmark(RunRequest(cases_path="ignored.jsonl"), settings)

    assert run.config["benchmark_org_id"] == "bench-org"
    assert _FakeClient.instances[0].org_id == run.config["benchmark_org_id"]
    assert _FakeClient.instances[0].ingest_calls == ["case-1"]
    assert _FakeClient.instances[0].chat_queries == ["What therapy is safest?"]
    assert run.cases[0].trace_id == "trace-case-1"
    assert run.cases[0].errors == []
    assert run.config["reranker_budget"] == 1
    assert run.diagnostics["setup_stats_before"]["index_provenance_valid"] is True
    assert run.diagnostics["request_stats"]["retries"] == 0


@pytest.mark.asyncio
async def test_run_benchmark_skips_reingest_when_benchmark_context_is_ready(monkeypatch, tmp_path):
    class _ReadyClient(_FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                **kwargs,
                storage_stats_payload={
                    "source_counts": {"CPG": 0, "EMR": 1, "LIT": 1},
                    "index_exists": True,
                    "index_manifest": {
                        "org_id": "bench-org",
                        "embedding_space": RuntimeSettings().active_embedding_space(),
                        "embedding_model": RuntimeSettings().active_embedding_model(),
                        "embedding_dim": RuntimeSettings().active_embedding_dimension(),
                    },
                    "index_provenance_valid": True,
                    "active_embedding_space": RuntimeSettings().active_embedding_space(),
                    "active_embedding_model": RuntimeSettings().active_embedding_model(),
                    "active_embedding_dim": RuntimeSettings().active_embedding_dimension(),
                },
            )

    _ReadyClient.instances = []
    case = BenchmarkCase(
        case_id="case-1",
        dataset="sample",
        query="What therapy is safest?",
        patient_id="patient-1",
        patient_context="Patient context that should already be present in the benchmark org.",
        gold_facets=[GoldFacet(facet_id="treatment", name="treatment", critical=True)],
    )

    monkeypatch.setattr("eval.app.runner.read_jsonl_cases", lambda path: [case])
    monkeypatch.setattr("eval.app.runner.MedSwinClient", _ReadyClient)

    settings = Settings(
        medswin_base_url="http://medswin.test",
        benchmark_org_id="bench-org",
        benchmark_user_id="bench-user",
        request_timeout_s=5.0,
        run_store_dir=str(tmp_path),
        max_cases_default=25,
    )

    run = await run_benchmark(RunRequest(cases_path="ignored.jsonl"), settings)

    assert run.diagnostics["case_context_ready"] is True
    assert _ReadyClient.instances[0].ingest_calls == []
    assert _ReadyClient.instances[0].chat_queries == ["What therapy is safest?"]
    assert run.cases[0].errors == []


def test_trace_completeness_accepts_runtime_plural_count_keys():
    response = {
        "answer": "Clinical decision support only.",
        "evidence_bundle": {"passages": [], "evidence_ledger": []},
        "policy_decision": {"passed": True},
        "facet_coverage": [],
        "citations": [],
        "evidence_ledger": [],
        "trace_id": "trace-1",
    }
    trace_summary = {
        "trace_id": "trace-1",
        "messages_count": 2,
        "tool_calls_count": 1,
        "sufficiency_checks_count": 1,
        "evidence_passages_count": 4,
    }

    assert trace_completeness(response, trace_summary) == 1.0


def test_groundedness_proxy_does_not_treat_contradictions_as_support():
    score, penalty = groundedness_proxy(
        {
            "answer": "Clinical decision support only.",
            "evidence_ledger": [
                {
                    "claim": "Avoid this therapy in the presence of severe allergy.",
                    "doc_id": "doc-1",
                    "polarity": "contradicts",
                }
            ],
            "citations": [{"doc_id": "doc-1"}],
        },
        {"doc-1"},
    )

    assert score == 0.0
    assert penalty == 1.0


def test_validate_setup_stats_rejects_foreign_or_stale_index():
    with pytest.raises(RuntimeError, match="provenance"):
        _validate_setup_stats(
            {
                "source_counts": {"CPG": 0, "EMR": 0, "LIT": 1},
                "index_exists": True,
                "index_manifest": {
                    "org_id": "other-org",
                    "embedding_space": RuntimeSettings().active_embedding_space(),
                    "embedding_model": RuntimeSettings().active_embedding_model(),
                    "embedding_dim": RuntimeSettings().active_embedding_dimension(),
                },
                "active_embedding_space": RuntimeSettings().active_embedding_space(),
                "active_embedding_model": RuntimeSettings().active_embedding_model(),
                "active_embedding_dim": RuntimeSettings().active_embedding_dimension(),
            },
            "bench-org",
        )


def test_validate_setup_stats_rejects_incomplete_lit_index():
    active_settings = RuntimeSettings()
    with pytest.raises(RuntimeError, match="LIT vectors|vector count"):
        _validate_setup_stats(
            {
                "source_counts": {"CPG": 0, "EMR": 0, "LIT": 10},
                "active_embeddings": 10,
                "index_exists": True,
                "index_manifest": {
                    "org_id": "bench-org",
                    "embedding_space": active_settings.active_embedding_space(),
                    "embedding_model": active_settings.active_embedding_model(),
                    "embedding_dim": active_settings.active_embedding_dimension(),
                    "total_vectors": 2,
                    "source_counts": {"CPG": 0, "EMR": 0, "LIT": 0},
                },
                "active_embedding_space": active_settings.active_embedding_space(),
                "active_embedding_model": active_settings.active_embedding_model(),
                "active_embedding_dim": active_settings.active_embedding_dimension(),
            },
            "bench-org",
        )


def test_audit_case_reports_qrel_availability_and_failure_bucket():
    case = BenchmarkCase(
        case_id="case-1",
        dataset="sample",
        query="Clinical question",
        gold_doc_ids=["gold-1"],
        gold_facets=[GoldFacet(facet_id="clinical", name="clinical", critical=True, gold_doc_ids=["gold-1"])],
    )
    audit = audit_case(
        case,
        {
            "answer": "Insufficient evidence.",
            "trace_id": "trace-1",
            "policy_decision": {"passed": False},
            "citations": [{"doc_id": "emr-1", "source_type": "EMR"}],
            "evidence_bundle": {
                "passages": [{"doc_id": "emr-1", "chunk_id": "emr-1_chunk_0", "source_type": "EMR"}],
                "evidence_ledger": [],
            },
        },
        {"messages_count": 1, "sufficiency_checks_count": 1},
        available_doc_ids={"gold-1"},
        indexed_doc_ids={"gold-1"},
    )

    assert audit.gold_available_in_corpus == 1
    assert audit.gold_available_in_index == 1
    assert audit.gold_available_but_not_retrieved is True
    assert audit.selected_source_counts["EMR"] == 1
    assert audit.failure_bucket == "no_literature_retrieved"


def test_audit_case_uses_case_qrels_when_facet_qrels_are_stale():
    case = BenchmarkCase(
        case_id="case-1",
        dataset="sample",
        query="Clinical question",
        gold_doc_ids=["gold-1"],
        gold_facets=[GoldFacet(facet_id="clinical", name="clinical", critical=True, gold_doc_ids=["stale-1"])],
    )
    audit = audit_case(
        case,
        {
            "answer": "Grounded answer.",
            "trace_id": "trace-1",
            "policy_decision": {"passed": True},
            "citations": [{"doc_id": "gold-1", "source_type": "LIT"}],
            "evidence_bundle": {
                "passages": [{"doc_id": "gold-1", "chunk_id": "gold-1_chunk_0", "source_type": "LIT"}],
                "evidence_ledger": [],
            },
        },
        {"messages_count": 1, "sufficiency_checks_count": 1},
    )

    assert audit.facet_recall == 1.0
    assert audit.critical_facet_recall == 1.0


def test_validate_qrel_coverage_rejects_missing_judged_pool():
    with pytest.raises(RuntimeError, match="qrel coverage"):
        _validate_qrel_coverage(
            {
                "gold_doc_count": 10,
                "gold_corpus_recall": 0.2,
                "gold_index_recall": 0.0,
                "missing_gold_doc_ids_sample": ["doc-1"],
            },
            min_corpus_recall=0.95,
            min_index_recall=0.95,
        )
