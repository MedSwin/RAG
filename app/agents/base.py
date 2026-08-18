"""Shared agent helpers for structured claim emission."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity
from app.schemas.evidence import CandidatePassage, EvidenceClaim
from app.schemas.facets import ClinicalFacet
from app.services.adapters.llm import LLMClient
from app.services.prompts.structured import extract_json_object
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
    blocks = []
    for passage in passages[:limit]:
        blocks.append(
            f"[{passage.chunk_id}|{passage.source_type.value}|doc={passage.doc_id}]\n{passage.text[:1200]}"
        )
    return "\n\n".join(blocks)


def parse_claims(
    agent_id: str,
    payload: Dict[str, Any],
    passages: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
) -> AgentClaimBatch:
    allowed = {p.chunk_id: p for p in passages}
    claims: List[EvidenceClaim] = []
    for item in payload.get("claims") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id not in allowed:
            continue
        passage = allowed[chunk_id]
        try:
            polarity = EvidencePolarity(item.get("polarity") or "supports")
        except ValueError:
            polarity = EvidencePolarity.SUPPORTS
        grade = evidence_grade_from_metadata(passage)
        claims.append(
            EvidenceClaim(
                facet=canonicalize_facet(str(item.get("facet") or "clinical_evidence"), facets),
                claim=str(item.get("claim") or "")[:500],
                polarity=polarity,
                chunk_id=chunk_id,
                confidence=float(item.get("confidence") or passage.calibrated_score or 0.5),
                evidence_grade=grade,
                calibrated_relevance=float(passage.calibrated_score or passage.rerank_score or 0.0),
                agent_id=agent_id,
                provenance={
                    "doc_id": passage.doc_id,
                    "section": passage.section,
                    "source_type": passage.source_type.value,
                },
            )
        )
    return AgentClaimBatch(
        agent_id=agent_id,
        claims=claims,
        notes=[str(n) for n in (payload.get("notes") or []) if n],
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
