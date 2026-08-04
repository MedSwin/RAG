"""Shared enumerations for MedSwin runtime artefacts."""

from enum import Enum


class SourceType(str, Enum):
    """Source type for documents and passages."""

    CPG = "CPG"
    EMR = "EMR"
    LIT = "LIT"
    SAFETY = "SAFETY"


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


class AgentRole(str, Enum):
    """Orchestrator-mediated specialist roles."""

    NORMALIZE = "normalize"
    EMR = "emr"
    GUIDELINE = "guideline"
    SAFETY = "safety"
    QUALITY = "quality"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"
    RETRIEVAL = "retrieval"
