"""Thin final-generation schema used only when SYSTEM_EVAL_MODE is enabled."""

SYSTEM_SUFFIX = """
For this benchmark request, additionally return `exact_answer` and
`ideal_answer`. `ideal_answer` must be a concise substantive answer no longer
than 200 words. Do not change retrieval, evidence selection, or policy.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "exact_answer": {},
        "ideal_answer": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "object"}},
        "uncertainty": {"type": "string"},
        "contraindications_risks": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "exact_answer", "ideal_answer", "evidence_used", "uncertainty", "contraindications_risks", "next_steps", "insufficient_evidence"],
    "additionalProperties": True,
}
