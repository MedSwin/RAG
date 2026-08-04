"""Models package - includes Pydantic MedSwin artifacts.

Model manager utilities should be imported from their concrete modules to avoid
pulling optional ML/runtime dependencies into lightweight app tests.
"""

from app.schemas import (
    QuerySpec,
    CandidatePassage,
    RerankScore,
    EvidenceBundle,
    EMRSummary,
    GuidelineSummary,
    SafetyReport,
    ChatResponse,
    AuditTrace,
    Session,
    Document,
    Chunk,
    SourceType,
    PolicyAction,
    ClinicalFacet,
)

__all__ = [
    "QuerySpec",
    "CandidatePassage",
    "RerankScore",
    "EvidenceBundle",
    "EMRSummary",
    "GuidelineSummary",
    "SafetyReport",
    "ChatResponse",
    "AuditTrace",
    "Session",
    "Document",
    "Chunk",
    "SourceType",
    "PolicyAction",
    "ClinicalFacet",
]
