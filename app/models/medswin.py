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


class QuerySpec(BaseModel):
    """Normalized query specification."""
    canonical_terms: List[str] = Field(default_factory=list)
    abbreviations: Dict[str, str] = Field(default_factory=dict)
    retrieval_hints: Dict[str, Any] = Field(default_factory=dict)
    specialty: Optional[str] = None
    medications: List[str] = Field(default_factory=list)
    labs: List[str] = Field(default_factory=list)


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
    citations: List[Dict[str, str]] = Field(default_factory=list)


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
    final_answer: Optional[str] = None
    citations: List[Dict[str, str]] = Field(default_factory=list)


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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # For BM25
    tokenized_text: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

