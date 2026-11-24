"""Session persistence schema."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Session(BaseModel):
    """Session model for MongoDB."""

    session_id: str
    user_id: str
    org_id: str
    patient_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
