"""Shared agent helpers for structured claim emission."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity
from app.schemas.evidence import CandidatePassage, EvidenceClaim
from app.schemas.facets import ClinicalFacet
from app.services.adapters.llm import LLMClient
from app.services.medswin.governance import redact_phi_text
from app.services.prompts.structured import _is_schema_fragment, extract_json_object, schema_instruction
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

# Identical leading system text for every specialist. Role text stays in the suffix
# so the five sequential calls share one KV-cache prefix.
SHARED_CLAIM_SYSTEM_PREFIX = (
    "You are a MedSwin specialist in a shared claim workflow. "
    "The shared evidence passages are identical for every specialist. "
    "Apply only the specialist role, query, target facets, and valid chunk_ids that follow the passages. "
    "Return JSON only."
)
SPECIALIST_SUFFIX_MARK = "\n\nSpecialist Instructions:\n"
_LOCAL_GENERATION_BACKENDS = {"medswin_local", "local_http"}
_COMPACT_PASSAGE_LIMIT = 4
_COMPACT_PASSAGE_CHARS = 280
_COMPACT_PASSAGE_BLOCK_CHARS = 1400


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


def _local_generation_window() -> bool:
    return settings.resolved_generation_backend() in _LOCAL_GENERATION_BACKENDS


def passage_priority_key(passage: CandidatePassage, index: int) -> tuple:
    """Rank contraindications and high-scoring passages ahead of filler."""
    safety = 1 if passage.source_type.value == "SAFETY" or float(passage.safety_score or 0.0) >= 0.3 else 0
    contradiction = float(passage.contradiction_score or 0.0)
    score = passage.calibrated_score
    if score is None:
        score = passage.rerank_score
    if score is None:
        score = passage.fusion_score
    if score is None:
        score = passage.dense_score
    if score is None:
        score = 0.0
    return (-int(safety), -float(contradiction), -float(score), index)


def prioritize_passages(passages: List[CandidatePassage]) -> List[CandidatePassage]:
    ordered = sorted(enumerate(passages), key=lambda item: passage_priority_key(item[1], item[0]))
    return [passage for _, passage in ordered]


def shared_claim_system_message() -> str:
    """Return the byte-stable system prefix, including the claim JSON schema."""
    if _local_generation_window():
        return (
            f"{SHARED_CLAIM_SYSTEM_PREFIX}\n"
            'Return one JSON object. Example: {"claims":[{"facet":"guideline_concordance",'
            '"claim":"...","polarity":"supports","chunk_id":"id#0","confidence":0.85}],"notes":[]}'
        )
    return f"{SHARED_CLAIM_SYSTEM_PREFIX}\n{schema_instruction(CLAIM_SCHEMA)}"


def passage_context(passages: List[CandidatePassage], limit: int = 12) -> str:
    """Render bounded specialist context.

    Normal application behavior retains the historical 12 × 1,200-character
    envelope. system-eval sets the same smaller envelope
    for *both* local MedSwin and Foundry GPT child processes so generator
    comparisons do not silently give the larger-context model more evidence.
    A local 2,048-token generator, with no explicit envelope, uses a shorter
    block and keeps safety passages at the front.
    """
    limit_set = "AGENT_PASSAGE_LIMIT" in os.environ
    chars_set = "AGENT_PASSAGE_MAX_CHARS" in os.environ
    compact = _local_generation_window() and not limit_set and not chars_set
    if limit_set:
        configured_limit = int(os.environ["AGENT_PASSAGE_LIMIT"])
    else:
        configured_limit = min(limit, _COMPACT_PASSAGE_LIMIT) if compact else limit
    if chars_set:
        configured_chars = int(os.environ["AGENT_PASSAGE_MAX_CHARS"])
    else:
        configured_chars = _COMPACT_PASSAGE_CHARS if compact else 1200
    configured_limit = max(1, configured_limit)
    configured_chars = max(128, configured_chars)
    selected = prioritize_passages(passages)
    total_cap = _COMPACT_PASSAGE_BLOCK_CHARS if compact else None
    blocks: List[str] = []
    for passage in selected:
        if len(blocks) >= configured_limit:
            break
        header = f"[{passage.chunk_id}|{passage.source_type.value}|doc={passage.doc_id}]"
        redacted = redact_phi_text(passage.text or "")
        body = redacted[:configured_chars]
        block = f"{header}\n{body}"
        if total_cap is not None:
            trial = "\n\n".join(blocks + [block])
            if len(trial) > total_cap:
                used = len("\n\n".join(blocks))
                separator = 2 if blocks else 0
                room = total_cap - used - separator - len(header) - 1
                if room <= 0:
                    break
                block = f"{header}\n{redacted[:room]}"
                trial = "\n\n".join(blocks + [block])
                if len(trial) > total_cap:
                    break
        blocks.append(block)
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


def _claim_user_message(
    role: str,
    task: str,
    shared: List[CandidatePassage],
    allowed: List[CandidatePassage],
) -> str:
    """Show only the passages this specialist may cite."""
    del shared
    allowed_ids = ", ".join(passage.chunk_id for passage in allowed)
    suffix = (
        f"{role.strip()}\n\n"
        f"{task.strip()}\n\n"
        f"Valid chunk_ids: {allowed_ids}\n"
        "Cite only a chunk_id from Valid chunk_ids. "
        "Keep the safety focus and the citation rule in this suffix."
    )
    return f"Shared Evidence Passages:\n{passage_context(allowed)}{SPECIALIST_SUFFIX_MARK}{suffix}"


async def call_claim_agent(
    client: LLMClient,
    agent_id: str,
    system: str,
    user: str,
    passages: List[CandidatePassage],
    facets: Optional[List[ClinicalFacet]] = None,
    shared_passages: Optional[List[CandidatePassage]] = None,
) -> AgentClaimBatch:
    """Call one specialist.

    ``passages`` is the allowed citation set for ``parse_claims``.
    ``shared_passages`` is the identical evidence prefix for every specialist.
    The role text, query, facets, and valid chunk ids stay after that prefix.
    """
    if not passages:
        return AgentClaimBatch(
            agent_id=agent_id,
            claims=[],
            notes=["This specialist had no passages to read."],
        )
    shared = passages if shared_passages is None else shared_passages
    try:
        local_cap = 384 if getattr(client, "backend", "") in {"medswin_local", "local_http"} else None
        response = await client.call_llm(
            [
                {"role": "system", "content": shared_claim_system_message()},
                {"role": "user", "content": _claim_user_message(system, user, shared, passages)},
            ],
            json_schema=CLAIM_SCHEMA,
            max_tokens=local_cap,
        )
        payload = extract_json_object(response["content"])
        if (
            not isinstance(payload, dict)
            or _is_schema_fragment(payload)
            or "claims" not in payload
        ):
            return AgentClaimBatch(
                agent_id=agent_id,
                claims=[],
                degraded=True,
                error="schema_echo_only",
            )
        raw_claims = payload.get("claims")
        offered = [item for item in raw_claims if isinstance(item, dict)] if isinstance(raw_claims, list) else []
        batch = parse_claims(agent_id, payload, passages, facets=facets)
        if offered and not batch.claims:
            batch.degraded = True
            batch.error = "uncited_claims"
        return batch
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s agent failed: %s", agent_id, exc)
        return AgentClaimBatch(
            agent_id=agent_id,
            claims=[],
            degraded=True,
            error=type(exc).__name__,
        )
