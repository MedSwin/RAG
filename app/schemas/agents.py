"""Agent output and claim-batch schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceClaim


class AgentClaimBatch(BaseModel):
    """Structured claim batch emitted by a specialist agent."""

    agent_id: str
    claims: List[EvidenceClaim] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    degraded: bool = False
    error: Optional[str] = None


class EMRSummary(BaseModel):
    """Structured patient state summary (legacy-compatible)."""

    patient_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    problems: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    labs: Dict[str, Any] = Field(default_factory=dict)
    contraindications_flags: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)


class GuidelineSummary(BaseModel):
    """Guideline synthesis with recommendations (legacy-compatible)."""

    recommendations: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    guideline_strength: Optional[str] = None
    guideline_grade: Optional[str] = None
    source_guidelines: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)


class SafetyReport(BaseModel):
    """Safety critique report (legacy-compatible)."""

    missing_evidence: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    unsafe_suggestions: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    requires_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Evidence-quality agent output."""

    graded_chunk_ids: List[str] = Field(default_factory=list)
    weak_provenance: List[str] = Field(default_factory=list)
    population_mismatch: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)


class CriticReport(BaseModel):
    """Contradiction critic output."""

    conflicts: List[str] = Field(default_factory=list)
    resolved: List[str] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)
