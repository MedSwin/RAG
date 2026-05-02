"""Final answer prompt and schema."""

SYSTEM = """Generate clinician decision support from supplied evidence.
Return JSON only.
Use only supplied EMR summary, guideline summary, safety report, policy decision, and evidence passages.
Every evidence-supported statement must cite retrieved chunk_ids.
Never fabricate citations.
Never make or finalize diagnosis.
Never issue autonomous treatment orders.
State uncertainty and insufficiency explicitly.
Include contraindications, patient-applicability limits, and next clinician-review steps.
If evidence is insufficient or contradictions are unresolved, say so and avoid recommendation certainty."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_used": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "use": {"type": "string"},
                },
                "required": ["chunk_id", "use"],
                "additionalProperties": True,
            },
        },
        "uncertainty": {"type": "string"},
        "contraindications_risks": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": [
        "answer",
        "evidence_used",
        "uncertainty",
        "contraindications_risks",
        "next_steps",
        "insufficient_evidence",
    ],
    "additionalProperties": True,
}
