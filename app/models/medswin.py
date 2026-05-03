"""Pydantic models for MedSwin typed artifacts."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Source type for documents."""
    CPG = "CPG"  # Clinical Practice Guidelines
    EMR = "EMR"  # Electronic Medical Records
    LIT = "LIT"  # Literature


class ClinicalScope(str, Enum):
    """Clinical output boundary enforced by the runtime."""
    CLINICIAN_CDS = "clinician_cds"
    DIFFERENTIAL_DX = "differential_dx"
    PATIENT_ADVICE = "patient_advice"


class EvidencePolarity(str, Enum):
    """How a passage or claim relates to a clinical facet."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    SAFETY = "safety"
    IRRELEVANT = "irrelevant"


class PolicyAction(str, Enum):
    """Deterministic policy action chosen after evidence review."""
    ACCEPT = "accept"
    RETRIEVE_MORE = "retrieve_more"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REQUIRE_CLARIFICATION = "require_clarification"


class ClinicalFacet(BaseModel):
    """Clinical evidence facet required for safe CDS synthesis."""
    name: str
    required: bool = True
    threshold: float = 0.70
    weight: float = 1.0
    source_policy: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class EvidenceGrade(BaseModel):
    """Evidence hierarchy metadata used by policy-aware selection."""
    label: str = "ungraded"
    score: float = 0.50
    source_reliability: float = 0.50
    rationale: Optional[str] = None


class EvidenceClaim(BaseModel):
    """Claim-level evidence emitted by retrieval or specialist agents."""
    facet: str
    claim: str
    polarity: EvidencePolarity = EvidencePolarity.SUPPORTS
    chunk_id: str
    confidence: float = 0.0
    evidence_grade: EvidenceGrade = Field(default_factory=EvidenceGrade)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class EvidenceLedgerEntry(BaseModel):
    """Auditable passage-level evidence ledger entry."""
    chunk_id: str
    doc_id: str
    source_type: SourceType
    agent_id: str = "retrieval"
    facets: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)
    calibrated_relevance: float = 0.0
    fusion_score: float = 0.0
    evidence_grade: EvidenceGrade = Field(default_factory=EvidenceGrade)
    safety_relevance: float = 0.0
    contradiction_risk: float = 0.0
    provenance: Dict[str, Any] = Field(default_factory=dict)


class FacetCoverage(BaseModel):
    """Noisy-OR facet coverage and uncertainty estimate."""
    facet: str
    required: bool = True
    threshold: float = 0.70
    coverage_probability: float = 0.0
    lower_confidence_bound: float = 0.0
    entropy: float = 0.0
    status: str = "missing"
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    contradicting_chunk_ids: List[str] = Field(default_factory=list)


class ContradictionPair(BaseModel):
    """High-grade incompatible evidence that must not be averaged away."""
    facet: str
    chunk_id_a: str
    chunk_id_b: str
    severity: str = "medium"
    reason: str
    resolved: bool = False
    adjudication: Optional[str] = None


class PolicyDecision(BaseModel):
    """Enterprise evidence policy decision for a retrieval iteration."""
    passed: bool
    action: PolicyAction
    reason: str
    iteration: int = 0
    clinical_scope: ClinicalScope = ClinicalScope.CLINICIAN_CDS
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    marginal_utility_per_token: float = 0.0
    unresolved_critical_conflicts: bool = False
    missing_facets: List[str] = Field(default_factory=list)
    retrieval_hints: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuerySpec(BaseModel):
    """Normalized query specification."""
    canonical_terms: List[str] = Field(default_factory=list)
    abbreviations: Dict[str, str] = Field(default_factory=dict)
    retrieval_hints: Dict[str, Any] = Field(default_factory=dict)
    specialty: Optional[str] = None
    medications: List[str] = Field(default_factory=list)
    labs: List[str] = Field(default_factory=list)
    facets: List[ClinicalFacet] = Field(default_factory=list)
    clinical_scope: ClinicalScope = ClinicalScope.CLINICIAN_CDS


class CandidatePassage(BaseModel):
    """Candidate passage from retrieval."""
    chunk_id: str
    doc_id: str
    source_type: SourceType
    text: str
    section: Optional[str] = None
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rerank_score: Optional[float] = None
    fusion_score: Optional[float] = None
    token_count: Optional[int] = None
    calibrated_score: Optional[float] = None
    recency_score: Optional[float] = None
    section_score: Optional[float] = None
    source_score: Optional[float] = None
    evidence_grade_score: Optional[float] = None
    noise_score: Optional[float] = None
    safety_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    facet_scores: Dict[str, float] = Field(default_factory=dict)
    selected_reason: Optional[str] = None


class RerankScore(BaseModel):
    """Reranker output with calibration."""
    chunk_id: str
    logit: Optional[float] = None
    p_hat: float  # Calibrated probability
    calibration_version: Optional[str] = None


class EvidenceBundle(BaseModel):
    """Selected evidence bundle under token budget."""
    passages: List[CandidatePassage]
    total_tokens: int
    cpg_count: int
    emr_count: int
    lit_count: int
    coverage_ratios: Dict[str, float] = Field(default_factory=dict)
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntry] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    policy_decision: Optional[PolicyDecision] = None


class EMRSummary(BaseModel):
    """Structured patient state summary."""
    patient_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    problems: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    labs: Dict[str, Any] = Field(default_factory=dict)
    contraindications_flags: List[str] = Field(default_factory=list)


class GuidelineSummary(BaseModel):
    """Guideline synthesis with recommendations."""
    recommendations: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    guideline_strength: Optional[str] = None
    guideline_grade: Optional[str] = None
    source_guidelines: List[str] = Field(default_factory=list)


class SafetyReport(BaseModel):
    """Safety critique report."""
    missing_evidence: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    unsafe_suggestions: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    requires_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)


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


class AgentMessage(BaseModel):
    """Agent message in trace."""
    role: str  # "user", "assistant", "system", "tool"
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


class Session(BaseModel):
    """Session model for MongoDB."""
    session_id: str
    user_id: str
    org_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document model for MongoDB."""
    doc_id: str
    source_type: SourceType
    title: str
    version: Optional[str] = None
    effective_date: Optional[datetime] = None
    patient_id: Optional[str] = None
    org_id: str
    tags: List[str] = Field(default_factory=list)
    source_reliability: float = 0.50
    evidence_grade: Optional[EvidenceGrade] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """Chunk model for MongoDB."""
    chunk_id: str
    doc_id: str
    source_type: SourceType
    text: str
    section: Optional[str] = None
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    patient_id: Optional[str] = None
    guideline_version: Optional[str] = None
    timestamp: Optional[datetime] = None
    org_id: str
    evidence_grade: Optional[EvidenceGrade] = None
    source_reliability: float = 0.50
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    embedding_space: Optional[str] = None
    embedding_updated_at: Optional[datetime] = None
    # For BM25
    tokenized_text: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
