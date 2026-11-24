"""Shared structured-output controls."""

import json
from typing import Any, Dict

JSON_SCHEMA_SYSTEM = "Return valid JSON matching this schema. Return JSON only. No markdown. No prose outside JSON."


def _example_hint(schema: Dict[str, Any]) -> str:
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return ""
    if "claims" in props:
        return (
            ' Example: {"claims":[{"facet":"guideline_concordance","claim":"...",'
            '"polarity":"supports","chunk_id":"id#0","confidence":0.85}],"notes":[]}'
        )
    if "canonical_terms" in props:
        return (
            ' Example: {"canonical_terms":["atrial fibrillation"],'
            '"clinical_scope":"clinician_cds","abbreviations":{},"facets":[]}'
        )
    if "answer" in props and "evidence_used" in props:
        return (
            ' Example: {"answer":"...","evidence_used":[{"chunk_id":"id#0","use":"supporting"}],'
            '"uncertainty":"low","contraindications_risks":[],"next_steps":[]}'
        )
    return ""


def schema_instruction(schema: Dict[str, Any]) -> str:
    """Build a compact schema instruction for providers without native schema mode."""
    return f"{JSON_SCHEMA_SYSTEM} Schema: {json.dumps(schema, separators=(',', ':'))}{_example_hint(schema)}"


def _is_schema_fragment(obj: Dict[str, Any]) -> bool:
    """True for an echoed JSON Schema node, not a model payload."""
    schema_keys = {
        "type",
        "properties",
        "items",
        "required",
        "description",
        "enum",
        "additionalProperties",
        "$schema",
        "$id",
    }
    if obj.get("type") in {"object", "array", "string", "number", "integer", "boolean"} and (
        "properties" in obj or "items" in obj or set(obj).issubset(schema_keys)
    ):
        return True
    if obj and all(isinstance(value, dict) and "type" in value and "claim" not in value for value in obj.values()):
        return True
    return False


def extract_json_object(content: str) -> Dict[str, Any]:
    """Extract a JSON object from plain, fenced, or prose-wrapped model output.

    Motivation vs Logic: MedSwin agents exchange governed typed artifacts, not
    narrative blobs. 7B local models may wrap JSON in brief prose or fenced
    blocks; scanning with JSONDecoder.raw_decode recovers valid dicts while
    skipping echoed JSON-Schema definitions.
    """
    text = (content or "").strip()
    if not text:
        return {}
    candidates = [text]
    if "```json" in text:
        candidates.insert(0, text.split("```json", 1)[1].split("```", 1)[0].strip())
    elif "```" in text:
        candidates.insert(0, text.split("```", 1)[1].split("```", 1)[0].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    fallback_dict: Dict[str, Any] | None = None
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        is_schema_echo = _is_schema_fragment(obj)
        if not is_schema_echo:
            return obj
        if fallback_dict is None:
            fallback_dict = obj

    if fallback_dict is not None:
        return fallback_dict
    return json.loads(text)
