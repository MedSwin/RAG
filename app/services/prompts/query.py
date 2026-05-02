"""Query normalization prompt and schema."""

SYSTEM = """Normalize clinical queries for retrieval.
Return JSON only.
Extract canonical medical terms, abbreviations, medications, labs, specialty, clinical scope, and retrieval hints.
Use conservative expansions; do not add diagnoses, treatments, or facts absent from the query.
Represent required evidence needs as facets with name, required, threshold, weight, source_policy, and keywords.
Prefer clinician_cds unless the query explicitly asks for differential_dx or patient_advice."""

SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_terms": {"type": "array", "items": {"type": "string"}},
        "abbreviations": {"type": "object"},
        "retrieval_hints": {"type": "object"},
        "specialty": {"type": "string"},
        "medications": {"type": "array", "items": {"type": "string"}},
        "labs": {"type": "array", "items": {"type": "string"}},
        "clinical_scope": {"type": "string"},
        "facets": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["canonical_terms", "abbreviations", "retrieval_hints"],
    "additionalProperties": True,
}
