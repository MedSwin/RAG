from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class GoldFacet(BaseModel):
    facet_id: str
    name: str
    weight: float = 1.0
    critical: bool = False
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_terms: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkCase(BaseModel):
    case_id: str
    dataset: str = "trec-cds-2016"
    query: str
    query_type: str | None = None
    patient_id: str | None = None
    patient_context: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_facets: list[GoldFacet] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    cases_path: str = "data/sample/cases.jsonl"
    max_cases: int | None = None
    max_concurrency: int = 1
    reranker_budget: int | None = None
    ingest_case_context: bool = True
    fetch_trace_summary: bool = True
    include_patient_context_in_query: bool = False
    source_policy: Literal["ANY", "CPG_ONLY", "EMR_ONLY", "LIT_ONLY"] = "ANY"
    guideline_only: bool = False
    min_evidence_grade: float = 0.3
    clinical_scope: str = "clinician_cds"


class CaseAudit(BaseModel):
    case_id: str
    dataset: str
    trace_id: str | None = None
    session_id: str | None = None
    policy_passed: bool | None = None
    degraded_mode: bool | None = None
    answer_chars: int = 0
    selected_doc_ids: list[str] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    selected_source_counts: dict[str, int] = Field(default_factory=dict)
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_available_in_corpus: int | None = None
    gold_available_in_index: int | None = None
    gold_available_but_not_retrieved: bool | None = None
    failure_bucket: str | None = None
    facet_recall: float = 0.0
    critical_facet_recall: float = 0.0
    evidence_doc_recall: float = 0.0
    citation_precision: float = 0.0
    trace_completeness: float = 0.0
    sufficiency_decision_score: float = 0.0
    groundedness_proxy: float = 0.0
    unsupported_penalty: float = 0.0
    unsafe_omission_penalty: float = 0.0
    msas: float = 0.0
    trace_rate_limit_stats: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class RunAudit(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset: str = "trec-cds-2016"
    num_cases: int = 0
    aggregate: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseAudit] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
