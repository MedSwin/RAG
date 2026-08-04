"""Evidence, passage, ledger, and bundle schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.enums import ClinicalScope, EvidencePolarity, PolicyAction, SourceType
from app.schemas.facets import ClinicalFacet, FacetCoverage


class EvidenceGrade(BaseModel):
    """Evidence hierarchy metadata used by policy-aware selection."""

    label: str = "ungraded"
    score: float = 0.50
    source_reliability: float = 0.50
    rationale: Optional[str] = None
    vector: Dict[str, float] = Field(default_factory=dict)


class EvidenceClaim(BaseModel):
    """Claim-level evidence emitted by retrieval or specialist agents."""

    facet: str
    claim: str
    polarity: EvidencePolarity = EvidencePolarity.SUPPORTS
    chunk_id: str
    confidence: float = 0.0
    evidence_grade: EvidenceGrade = Field(default_factory=EvidenceGrade)
    calibrated_relevance: float = 0.0
    agent_id: str = "retrieval"
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


class ContradictionPair(BaseModel):
    """High-grade incompatible evidence that must not be averaged away."""

    facet: str
    chunk_id_a: str
    chunk_id_b: str
    severity: str = "medium"
    reason: str
    resolved: bool = False
    adjudication: Optional[str] = None
    grade_a: float = 0.0
    grade_b: float = 0.0


class ContradictionLedger(BaseModel):
    """Audit artefact: contradiction ledger."""

    pairs: List[ContradictionPair] = Field(default_factory=list)
    unresolved_high: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    routed_agent: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SufficiencyDecision(BaseModel):
    """Audit artefact: final sufficiency decision."""

    passed: bool
    action: PolicyAction
    reason: str
    iteration: int = 0
    missing_facets: List[str] = Field(default_factory=list)
    unresolved_critical_conflicts: bool = False
    marginal_utility_per_token: float = 0.0
    routed_agent: Optional[str] = None


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
    agent_scores: Dict[str, float] = Field(default_factory=dict)
    retrieved_by: List[str] = Field(default_factory=list)


class RerankScore(BaseModel):
    """Reranker output with calibration."""

    chunk_id: str
    logit: Optional[float] = None
    p_hat: float
    calibration_version: Optional[str] = None


class EvidenceBundle(BaseModel):
    """Selected evidence bundle under token budget."""

    passages: List[CandidatePassage]
    total_tokens: int
    cpg_count: int
    emr_count: int
    lit_count: int
    safety_count: int = 0
    coverage_ratios: Dict[str, float] = Field(default_factory=dict)
    facet_coverage: List[FacetCoverage] = Field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntry] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    policy_decision: Optional[PolicyDecision] = None


class AnswerProvenance(BaseModel):
    """Audit artefact: links answer statements to accepted passages."""

    statements: List[Dict[str, Any]] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    rejected_chunk_ids: List[str] = Field(default_factory=list)
