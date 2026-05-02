"""EMR summarization prompt and schema."""

SYSTEM = """Extract patient state from EMR passages.
Return JSON only.
Use only supplied passages.
Preserve uncertainty, dates, values, units, allergies, medications, labs, vitals, comorbidities, and contraindication flags.
Do not infer missing history.
Do not recommend treatment or make a diagnosis.
Every clinically important item should remain traceable to supplied text."""

SCHEMA = {
    "type": "object",
    "properties": {
        "timeline": {"type": "array"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "medications": {"type": "array", "items": {"type": "string"}},
        "allergies": {"type": "array", "items": {"type": "string"}},
        "vitals": {"type": "object"},
        "labs": {"type": "object"},
        "contraindications_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["timeline", "problems", "medications", "allergies", "vitals", "labs", "contraindications_flags"],
    "additionalProperties": True,
}
