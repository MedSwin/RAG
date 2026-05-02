"""Safety critique prompt and schema."""

SYSTEM = """Critique clinical decision-support safety.
Return JSON only.
Use only query, EMR summary, guideline summary, policy artifacts, and supplied evidence metadata.
Identify missing evidence, conflicts, unsafe suggestions, contraindications, citation gaps, and boundary violations.
Require clarification when patient-specific evidence is needed but absent.
Do not make or finalize diagnoses.
Do not resolve high-severity contradictions without explicit evidence."""

SCHEMA = {
    "type": "object",
    "properties": {
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "unsafe_suggestions": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
        "requires_clarification": {"type": "boolean"},
        "clarification_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "missing_evidence",
        "conflicts",
        "unsafe_suggestions",
        "insufficient_evidence",
        "requires_clarification",
        "clarification_questions",
    ],
    "additionalProperties": True,
}
