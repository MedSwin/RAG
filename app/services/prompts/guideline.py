"""Guideline synthesis prompt and schema."""

SYSTEM = """Extract guideline evidence from CPG passages.
Return JSON only.
Use only supplied passages.
Extract recommendations, contraindications, eligible populations, excluded populations, strength, grade, version/date, and source guideline identifiers.
Cite chunk_ids inside recommendation text when available.
Do not invent guideline strength, dosing, or contraindications.
Mark missing or unclear grade as unknown."""

SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "contraindications": {"type": "array", "items": {"type": "string"}},
        "guideline_strength": {"type": "string"},
        "guideline_grade": {"type": "string"},
        "source_guidelines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["recommendations", "contraindications", "source_guidelines"],
    "additionalProperties": True,
}
