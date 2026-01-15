"""Models package - includes Pydantic models and model management."""

from app.models.medswin import (
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
)
from app.models.manager import ModelManager
from app.models.download import ModelDownloadService

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
    "ModelManager",
    "ModelDownloadService",
]

