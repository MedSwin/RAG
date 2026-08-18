"""Export normative audit JSON schemas for ChatResponse and AuditTrace."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.traces import AuditTrace, ChatResponse


def write_audit_schema(path: Path | None = None) -> Path:
    target = path or Path("app/schemas/audit.json")
    payload = {
        "title": "MedSwin runtime audit artefacts",
        "chat_response": ChatResponse.model_json_schema(),
        "audit_trace": AuditTrace.model_json_schema(),
    }
    target.write_text(json.dumps(payload, indent=2))
    return target
