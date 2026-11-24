"""Final synthesis agent — only after sufficiency gate accepts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.agents.base import _is_prompt_caution, prioritize_passages
from app.schemas.evidence import AnswerArtifact, AnswerProvenance, CandidatePassage, EvidenceBundle
from app.schemas.facets import FacetCoverage
from app.services.adapters.llm import LLMClient
from app.services.medswin.governance import ensure_cds_language, lead_with_caution, redact_phi_text
from app.services.prompts import answer as answer_prompt
from app.services.prompts.structured import extract_json_object


_LOCAL_BACKENDS = {"medswin_local", "local_http"}
_INLINE_CITATION_USE = "inline cited supporting evidence"
_CHUNK_TOKEN = re.compile(r"[A-Za-z0-9_.:\-]+")


def _text_items(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def _bounded_local_evidence(passages: Sequence[CandidatePassage]) -> tuple[str, List[str]]:
    """Keep a 2,048-token synthesis prompt inside about 2,800 evidence characters."""
    ordered = prioritize_passages(list(passages))
    cautions = [passage for passage in ordered if _is_prompt_caution(passage)]
    others = [passage for passage in ordered if passage not in cautions]
    blocks: List[str] = []
    shown: List[str] = []
    for passage in cautions + others:
        if len(blocks) >= 12:
            break
        if len(blocks) >= 6 and not _is_prompt_caution(passage):
            break
        body = lead_with_caution(redact_phi_text(passage.text or ""))[:450]
        block = f"[{passage.chunk_id}] {body}"
        trial = "\n\n".join(blocks + [block])
        if len(trial) > 2800:
            used = len("\n\n".join(blocks))
            separator = 2 if blocks else 0
            room = 2800 - used - separator - len(f"[{passage.chunk_id}] ")
            if room <= 0:
                break
            block = f"[{passage.chunk_id}] {lead_with_caution(redact_phi_text(passage.text or ''))[:room]}"
            trial = "\n\n".join(blocks + [block])
            if len(trial) > 2800:
                break
        blocks.append(block)
        shown.append(passage.chunk_id)
    return "\n\n".join(blocks), shown


def _valid_chunk_id_line(passages: Sequence[CandidatePassage], limit: int = 500) -> str:
    shown: List[str] = []
    text = ""
    seen = set()
    for passage in prioritize_passages(list(passages)):
        chunk_id = passage.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        addition = (", " if shown else "") + chunk_id
        if shown and len(text) + len(addition) > limit:
            break
        shown.append(chunk_id)
        text += addition
    return text


def _inline_chunk_ids(answer: str, allowed: set[str]) -> List[str]:
    """Read bracketed chunk ids in appearance order. Ignore bare tokens."""
    if not answer or not allowed:
        return []
    hits: List[tuple[int, str]] = []
    for match in re.finditer(r"\[([^\[\]]+)\]", answer):
        inner = match.group(1).strip()
        if inner in allowed:
            hits.append((match.start(), inner))
            continue
        for token in _CHUNK_TOKEN.findall(inner):
            if token in allowed:
                hits.append((match.start(), token))
    hits.sort(key=lambda item: (item[0], item[1]))
    ordered: List[str] = []
    seen = set()
    for _, chunk_id in hits:
        if chunk_id in allowed and chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


class SynthesisAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def _render(
        self,
        answer_data: Dict[str, Any],
        evidence_bundle: EvidenceBundle,
        shown_ids: Optional[set[str]] = None,
    ) -> tuple[str, AnswerProvenance]:
        bundle_ids = {p.chunk_id for p in evidence_bundle.passages}
        allowed = bundle_ids if shown_ids is None else bundle_ids & set(shown_ids)
        cited = []
        rejected = []
        statements = []
        raw_evidence = answer_data.get("evidence_used") if isinstance(answer_data, dict) else []
        if not isinstance(raw_evidence, list):
            raw_evidence = []
        for item in raw_evidence:
            if isinstance(item, dict):
                chunk_id = str(item.get("chunk_id") or "").strip()
                use = str(item.get("use") or "supporting evidence")
            elif isinstance(item, str):
                chunk_id = item.strip()
                use = "supporting evidence"
            else:
                continue
            if chunk_id in allowed:
                cited.append(chunk_id)
                statements.append({"chunk_id": chunk_id, "use": use})
            elif chunk_id:
                rejected.append(chunk_id)
        if not cited:
            answer = str(answer_data.get("answer") or "")
            for chunk_id in _inline_chunk_ids(answer, allowed):
                cited.append(chunk_id)
                statements.append({"chunk_id": chunk_id, "use": _INLINE_CITATION_USE})
        sections = [str(answer_data.get("answer", "")).strip()]
        if statements:
            sections.append("Evidence used:\n" + "\n".join(f"- {s['chunk_id']}: {s['use']}" for s in statements))
        else:
            sections.append("Evidence used: No valid retrieved chunk citations were provided by the model.")
        if answer_data.get("uncertainty"):
            sections.append(f"Uncertainty: {answer_data['uncertainty']}")
        risks = _text_items(answer_data.get("contraindications_risks"))
        if risks:
            sections.append("Contraindications/risks:\n" + "\n".join(f"- {item}" for item in risks))
        next_steps = _text_items(answer_data.get("next_steps"))
        if next_steps:
            sections.append("Next steps:\n" + "\n".join(f"- {item}" for item in next_steps))
        answer = ensure_cds_language("\n\n".join(part for part in sections if part))
        return answer, AnswerProvenance(
            statements=statements,
            cited_chunk_ids=cited,
            rejected_chunk_ids=rejected,
        )

    async def synthesize(
        self,
        query: str,
        evidence_bundle: EvidenceBundle,
        facet_coverage: Optional[List[FacetCoverage]] = None,
        safety_notes: Optional[List[str]] = None,
    ) -> tuple[str, AnswerProvenance]:
        rendered, provenance, _artifact = await self.synthesize_with_artifact(
            query, evidence_bundle, facet_coverage=facet_coverage, safety_notes=safety_notes
        )
        return rendered, provenance

    async def synthesize_with_artifact(
        self,
        query: str,
        evidence_bundle: EvidenceBundle,
        facet_coverage: Optional[List[FacetCoverage]] = None,
        safety_notes: Optional[List[str]] = None,
        benchmark: bool = False,
    ) -> tuple[str, AnswerProvenance, AnswerArtifact]:
        coverage_text = "\n".join(
            f"- {row.facet}: {row.status} (LCB={row.lower_confidence_bound:.2f})"
            for row in (facet_coverage or evidence_bundle.facet_coverage)
        )
        backend = getattr(self.client, "backend", "")
        shown_ids: Optional[set[str]] = None
        if backend in _LOCAL_BACKENDS:
            evidence_text, shown = _bounded_local_evidence(evidence_bundle.passages)
            shown_ids = set(shown)
            shown_passages = [passage for passage in evidence_bundle.passages if passage.chunk_id in shown_ids]
            notes = "; ".join(str(note) for note in (safety_notes or []) if str(note or "").strip()) or "none"
            user_content = (
                f"Query: {query}\n\nFacet coverage:\n{coverage_text}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Valid chunk_ids: {_valid_chunk_id_line(shown_passages)}\n"
                f"Safety notes: {notes}\n\n"
                "Provide a clinician decision-support answer with answer, evidence_used "
                "(chunk_ids only from Valid chunk_ids), uncertainty, contraindications_risks, next_steps. "
                "Do not present autonomous diagnosis or treatment orders."
            )
        else:
            evidence_text = "\n\n".join(
                f"[{p.chunk_id}] {lead_with_caution(redact_phi_text(p.text or ''))}"
                for p in prioritize_passages(list(evidence_bundle.passages))
            )
            user_content = (
                f"Query: {query}\n\nFacet coverage:\n{coverage_text}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Safety notes: {safety_notes or []}\n\n"
                "Provide a clinician decision-support answer with answer, evidence_used "
                "(chunk_ids only from evidence), uncertainty, contraindications_risks, next_steps. "
                "Do not present autonomous diagnosis or treatment orders."
            )
        messages = [
            {"role": "system", "content": answer_prompt.SYSTEM},
            {"role": "user", "content": user_content},
        ]
        schema = answer_prompt.SCHEMA
        if benchmark:
            from app.services.prompts import benchmark as benchmark_prompt
            messages[-1]["content"] += benchmark_prompt.SYSTEM_SUFFIX
            schema = benchmark_prompt.SCHEMA
        response = await self.client.call_llm(messages, json_schema=schema)
        raw_content = str(response.get("content") or "").strip()
        try:
            answer_data = extract_json_object(raw_content)
        except Exception:
            if backend not in _LOCAL_BACKENDS:
                raise
            answer_data = {}
        if backend in _LOCAL_BACKENDS and (
            not isinstance(answer_data, dict) or not str(answer_data.get("answer") or "").strip()
        ):
            answer_data = {
                "answer": "The model did not return a structured clinical answer.",
                "insufficient_evidence": True,
            }
        rendered, provenance = self._render(answer_data, evidence_bundle, shown_ids=shown_ids)
        ideal = " ".join(str(answer_data.get("ideal_answer") or answer_data.get("answer") or "").split()[:200]) if benchmark else None
        artifact = AnswerArtifact(
            content=str(answer_data.get("answer") or ""),
            rendered_text=rendered,
            exact_answer=answer_data.get("exact_answer") if benchmark else None,
            ideal_answer=ideal,
            cited_chunk_ids=provenance.cited_chunk_ids,
            abstained=bool(answer_data.get("insufficient_evidence", False)),
        )
        return rendered, provenance, artifact
