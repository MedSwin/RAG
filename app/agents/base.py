"""Shared agent helpers for structured claim emission."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity
from app.schemas.evidence import CandidatePassage, EvidenceClaim
from app.schemas.facets import ClinicalFacet
from app.services.adapters.llm import LLMClient
from app.services.prompts.structured import extract_json_object
from app.scoring.calibrate import first_present
from app.scoring.hierarchy import evidence_grade_from_metadata


_FACET_ALIASES = {
    "contraindication": "safety_contraindications",
    "contraindications": "safety_contraindications",
    "safety": "safety_contraindications",
    "guideline": "guideline_concordance",
    "recommendation": "guideline_concordance",
    "patient": "patient_applicability",
    "applicability": "patient_applicability",
    "quality": "evidence_quality",
    "grade": "evidence_quality",
}


def canonicalize_facet(name: str, facets: Optional[List[ClinicalFacet]] = None) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "clinical_evidence"
    names = [facet.name for facet in (facets or [])]
    if raw in names:
        return raw
    lowered = raw.lower().replace(" ", "_")
    for facet_name in names:
        if facet_name == lowered or facet_name in lowered or lowered in facet_name:
            return facet_name
    for key, target in _FACET_ALIASES.items():
        if key in lowered and (not names or target in names):
            return target
    return lowered


logger = logging.getLogger(__name__)


CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "facet": {"type": "string"},
                    "claim": {"type": "string"},
                    "polarity": {
                        "type": "string",
                        "enum": ["supports", "contradicts", "qualifies", "safety", "irrelevant"],
                    },
                    "chunk_id": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["facet", "claim", "chunk_id"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims"],
}


def passage_context(passages: List[CandidatePassage], limit: int = 12) -> str:
    """Render bounded specialist context.

    Normal application behavior retains the historical 12 × 1,200-character
    envelope. system-eval sets the same smaller envelope
    for *both* local MedSwin and Foundry GPT child processes so generator
    comparisons do not silently give the larger-context model more evidence.
    """
    configured_limit = int(os.getenv("AGENT_PASSAGE_LIMIT", str(limit)))
    configured_chars = int(os.getenv("AGENT_PASSAGE_MAX_CHARS", "1200"))
    configured_limit = max(1, configured_limit)
    configured_chars = max(128, configured_chars)
    blocks = []
    for passage in passages[:configured_limit]:
        blocks.append(
            f"[{passage.chunk_id}|{passage.source_type.value}|doc={passage.doc_id}]\n"
            f"{passage.text[:configured_chars]}"
        )
    return "\n\n".join(blocks)


def _finite_unit_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def parse_claims(
    agent_id: str,
    payload: Dict[str, Any],
    passages: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
) -> AgentClaimBatch:
    allowed = {p.chunk_id: p for p in passages}
    claims: List[EvidenceClaim] = []
    raw_claims = payload.get("claims") if isinstance(payload, dict) else []
    if not isinstance(raw_claims, list):
        raw_claims = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id not in allowed:
            continue
        passage = allowed[chunk_id]
        try:
            polarity = EvidencePolarity(item.get("polarity") or "supports")
        except ValueError:
            polarity = EvidencePolarity.SUPPORTS
        grade = evidence_grade_from_metadata(passage)
        parsed_confidence = _finite_unit_score(item.get("confidence"))
        fallback_confidence = _finite_unit_score(passage.calibrated_score)
        confidence = first_present(parsed_confidence, fallback_confidence, default=0.5)
        claims.append(
            EvidenceClaim(
                facet=canonicalize_facet(str(item.get("facet") or "clinical_evidence"), facets),
                claim=str(item.get("claim") or "")[:500],
                polarity=polarity,
                chunk_id=chunk_id,
                confidence=float(confidence),
                evidence_grade=grade,
                calibrated_relevance=float(first_present(passage.calibrated_score, passage.rerank_score, default=0.0)),
                agent_id=agent_id,
                provenance={
                    "doc_id": passage.doc_id,
                    "section": passage.section,
                    "source_type": passage.source_type.value,
                },
            )
        )
    raw_notes = payload.get("notes") if isinstance(payload, dict) else []
    if isinstance(raw_notes, str):
        notes = [raw_notes] if raw_notes.strip() else []
    elif isinstance(raw_notes, list):
        notes = [str(n) for n in raw_notes if n]
    else:
        notes = []
    return AgentClaimBatch(
        agent_id=agent_id,
        claims=claims,
        notes=notes,
    )


async def call_claim_agent(
    client: LLMClient,
    agent_id: str,
    system: str,
    user: str,
    passages: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
) -> AgentClaimBatch:
    try:
        response = await client.call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            json_schema=CLAIM_SCHEMA,
        )
        payload = extract_json_object(response["content"])
        batch = parse_claims(agent_id, payload, passages, facets=facets)
        return batch
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s agent failed: %s", agent_id, exc)
        return AgentClaimBatch(agent_id=agent_id, claims=[], degraded=True, error=str(exc))
