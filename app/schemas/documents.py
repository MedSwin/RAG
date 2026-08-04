"""Document and chunk persistence schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.enums import SourceType
from app.schemas.evidence import EvidenceGrade


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
    tokenized_text: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
