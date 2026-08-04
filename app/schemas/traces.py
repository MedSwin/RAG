"""Audit trace and chat response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.agents import EMRSummary, GuidelineSummary, SafetyReport
from app.schemas.enums import PolicyAction
from app.schemas.evidence import (
    AnswerProvenance,
    ContradictionLedger,
    ContradictionPair,
    EvidenceBundle,
    EvidenceLedgerEntry,
    PolicyDecision,
    SufficiencyDecision,
)
from app.schemas.facets import FacetCoverage, FacetMatrix


class AgentMessage(BaseModel):
    """Agent message in trace."""

    role: str
    agent_id: Optional[str] = None
    model_endpoint: Optional[str] = None
    content: str
    token_count: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolCall(BaseModel):
    """Tool call in trace."""

    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SufficiencyCheck(BaseModel):
    """Evidence sufficiency check result."""

    iteration: int
    kappa_cpg: float
    kappa_emr: float
    mean_confidence: float
    passed: bool
    action_taken: Optional[str] = None
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    contradiction_count: int = 0
    missing_facets: List[str] = Field(default_factory=list)
    marginal_utility_per_token: float = 0.0
    policy_decision: Optional[PolicyDecision] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RetrievalTrace(BaseModel):
    """Audit artefact: which agent retrieved each passage."""

    iteration: int = 0
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    dense_count: int = 0
    lexical_count: int = 0
    union_count: int = 0
    hints: Dict[str, Any] = Field(default_factory=dict)


class RerankTrace(BaseModel):
    """Audit artefact: calibrated relevance and select/reject."""

    iteration: int = 0
    scores: List[Dict[str, Any]] = Field(default_factory=list)
    calibration_version: Optional[str] = None
    selected_chunk_ids: List[str] = Field(default_factory=list)
    rejected_chunk_ids: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Final chat response with provenance."""

    answer: str
    evidence_bundle: EvidenceBundle
    safety_notes: Optional[str] = None
    trace_id: str
    degraded_mode: Dict[str, bool] = Field(default_factory=dict)
    uncertainty_level: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    policy_decision: Optional[PolicyDecision] = None
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntry] = Field(default_factory=list)
    sufficiency_decision: Optional[SufficiencyDecision] = None
    facet_matrix: Optional[FacetMatrix] = None
    contradiction_ledger: Optional[ContradictionLedger] = None
    answer_provenance: Optional[AnswerProvenance] = None
    retrieval_traces: List[RetrievalTrace] = Field(default_factory=list)
    rerank_traces: List[RerankTrace] = Field(default_factory=list)


class AuditTrace(BaseModel):
    """Full audit trace for a request."""

    trace_id: str
    session_id: str
    user_id: str
    org_id: str
    query: str
    patient_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    messages: List[AgentMessage] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    evidence_bundle: Optional[EvidenceBundle] = None
    sufficiency_checks: List[SufficiencyCheck] = Field(default_factory=list)
    policy_decisions: List[PolicyDecision] = Field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntry] = Field(default_factory=list)
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    final_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_traces: List[RetrievalTrace] = Field(default_factory=list)
    rerank_traces: List[RerankTrace] = Field(default_factory=list)
    facet_matrix: Optional[FacetMatrix] = None
    contradiction_ledger: Optional[ContradictionLedger] = None
    sufficiency_decision: Optional[SufficiencyDecision] = None
    answer_provenance: Optional[AnswerProvenance] = None
    emr_summary: Optional[EMRSummary] = None
    guideline_summary: Optional[GuidelineSummary] = None
    safety_report: Optional[SafetyReport] = None
    degraded_mode: Dict[str, bool] = Field(default_factory=dict)
