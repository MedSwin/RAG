from __future__ import annotations

from typing import Any


# Motivation vs Logic: benchmark facet templates must be shared by the eval
# harness and the live MedSwin policy engine so the benchmark does not drift
# from the scoring logic it is measuring.
BENCHMARK_FACET_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "diagnosis": [
        {
            "facet_id": "dx",
            "name": "diagnosis evidence",
            "weight": 1.0,
            "critical": True,
            "source_policy": "LIT",
            "keywords": ["diagnosis", "symptom", "findings", "assessment", "differential", "presentation"],
        },
        {
            "facet_id": "patient_fit",
            "name": "patient applicability",
            "weight": 1.0,
            "critical": True,
            "source_policy": "EMR",
            "keywords": ["patient", "history", "pmh", "age", "male", "female", "comorbidity", "medication"],
        },
    ],
    "test": [
        {
            "facet_id": "test_indication",
            "name": "test indication",
            "weight": 1.0,
            "critical": True,
            "source_policy": "LIT",
            "keywords": ["test", "indication", "diagnostic", "screening", "workup", "evaluation"],
        },
        {
            "facet_id": "patient_fit",
            "name": "patient applicability",
            "weight": 1.0,
            "critical": True,
            "source_policy": "EMR",
            "keywords": ["patient", "history", "pmh", "age", "pregnancy", "comorbidity", "allergy"],
        },
        {
            "facet_id": "risk",
            "name": "test risks or limitations",
            "weight": 0.75,
            "critical": False,
            "source_policy": "LIT",
            "keywords": ["risk", "limitation", "contraindication", "false positive", "false negative", "harm"],
        },
    ],
    "treatment": [
        {
            "facet_id": "treatment",
            "name": "treatment recommendation evidence",
            "weight": 1.0,
            "critical": True,
            "source_policy": "LIT",
            "keywords": ["treatment", "recommendation", "management", "therapy", "dose", "intervention"],
        },
        {
            "facet_id": "safety",
            "name": "safety contraindications or adverse risks",
            "weight": 1.2,
            "critical": True,
            "source_policy": "LIT",
            "keywords": ["safety", "contraindication", "adverse", "risk", "avoid", "interaction"],
        },
        {
            "facet_id": "patient_fit",
            "name": "patient applicability",
            "weight": 1.0,
            "critical": True,
            "source_policy": "EMR",
            "keywords": ["patient", "history", "pmh", "age", "comorbidity", "allergy", "medication"],
        },
    ],
}


def benchmark_facet_templates(query_type: str | None) -> list[dict[str, Any]]:
    templates = BENCHMARK_FACET_TEMPLATES.get((query_type or "").lower())
    if templates:
        return [dict(item) for item in templates]
    return [
        {
            "facet_id": "clinical_evidence",
            "name": "clinically relevant evidence",
            "weight": 1.0,
            "critical": True,
            "source_policy": "LIT",
            "keywords": ["evidence", "finding", "diagnosis", "treatment", "management"],
        },
        {
            "facet_id": "patient_fit",
            "name": "patient applicability",
            "weight": 1.0,
            "critical": True,
            "source_policy": "EMR",
            "keywords": ["patient", "history", "pmh", "age", "comorbidity", "allergy", "medication"],
        },
    ]


def benchmark_required_facets(query_type: str | None, facets: list[Any]) -> list[dict[str, Any]]:
    template_map = {item["name"]: item for item in benchmark_facet_templates(query_type)}
    global_template_map = {
        item["name"]: item
        for templates in BENCHMARK_FACET_TEMPLATES.values()
        for item in templates
    }
    required: list[dict[str, Any]] = []
    for facet in facets:
        payload = facet.model_dump() if hasattr(facet, "model_dump") else dict(facet)
        template = template_map.get(payload.get("name")) or global_template_map.get(payload.get("name"))
        if template:
            for key in ("source_policy", "keywords", "weight", "critical"):
                payload.setdefault(key, template.get(key))
        required.append(payload)
    return required

