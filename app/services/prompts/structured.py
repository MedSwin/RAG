"""Shared structured-output controls."""

import json
from typing import Any, Dict

JSON_SCHEMA_SYSTEM = "Return valid JSON matching this schema. Return JSON only. No markdown. No prose outside JSON."


def schema_instruction(schema: Dict[str, Any]) -> str:
    """Build a compact schema instruction for providers without native schema mode."""
    return f"{JSON_SCHEMA_SYSTEM} Schema: {json.dumps(schema, separators=(',', ':'))}"


def extract_json_object(content: str) -> Dict[str, Any]:
    """Extract a JSON object from plain or fenced model output.

    Motivation vs Logic: MedSwin agents exchange governed typed artifacts, not
    narrative blobs. A single parser keeps prompt compliance, fallback parsing,
    and tests consistent across all agent roles.
    """
    text = (content or "").strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    if not text:
        return {}
    return json.loads(text)
