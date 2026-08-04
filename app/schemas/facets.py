"""Clinical facet and coverage schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ClinicalFacet(BaseModel):
    """Clinical evidence facet required for safe CDS synthesis."""

    name: str
    required: bool = True
    threshold: float = 0.70
    weight: float = 1.0
    source_policy: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


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


class FacetMatrix(BaseModel):
    """Audit artefact: facet-coverage matrix for a query."""

    query: str = ""
    iteration: int = 0
    rows: List[FacetCoverage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
