"""Final synthesis agent — only after sufficiency gate accepts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.schemas.evidence import AnswerArtifact, AnswerProvenance, EvidenceBundle
from app.schemas.facets import FacetCoverage
from app.services.adapters.llm import LLMClient
from app.services.medswin.governance import ensure_cds_language
from app.services.prompts import answer as answer_prompt
from app.services.prompts.structured import extract_json_object


def _text_items(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


class SynthesisAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def _render(self, answer_data: Dict[str, Any], evidence_bundle: EvidenceBundle) -> tuple[str, AnswerProvenance]:
        allowed = {p.chunk_id for p in evidence_bundle.passages}
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
        evidence_text = "\n\n".join(f"[{p.chunk_id}] {p.text}" for p in evidence_bundle.passages)
        coverage_text = "\n".join(
            f"- {row.facet}: {row.status} (LCB={row.lower_confidence_bound:.2f})"
            for row in (facet_coverage or evidence_bundle.facet_coverage)
        )
        messages = [
            {"role": "system", "content": answer_prompt.SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nFacet coverage:\n{coverage_text}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    f"Safety notes: {safety_notes or []}\n\n"
                    "Provide a clinician decision-support answer with answer, evidence_used "
                    "(chunk_ids only from evidence), uncertainty, contraindications_risks, next_steps. "
                    "Do not present autonomous diagnosis or treatment orders."
                ),
            },
        ]
        schema = answer_prompt.SCHEMA
        if benchmark:
            from app.services.prompts import benchmark as benchmark_prompt
            messages[-1]["content"] += benchmark_prompt.SYSTEM_SUFFIX
            schema = benchmark_prompt.SCHEMA
        response = await self.client.call_llm(messages, json_schema=schema)
        answer_data = extract_json_object(response["content"])
        rendered, provenance = self._render(answer_data, evidence_bundle)
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
