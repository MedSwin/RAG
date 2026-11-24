"""Query normalisation → QuerySpec + facets."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.schemas.enums import ClinicalScope
from app.schemas.evidence import QuerySpec
from app.schemas.facets import ClinicalFacet
from app.schemas.traces import AgentMessage, AuditTrace
from app.services.adapters.llm import LLMClient
from app.services.prompts import query as query_prompt
from app.services.prompts.structured import extract_json_object

logger = logging.getLogger(__name__)

try:
    from facets import benchmark_required_facets
except Exception:  # noqa: BLE001
    def benchmark_required_facets(_case, explicit):
        return explicit


class QueryNormalizer:
    def __init__(self, client: LLMClient):
        self.client = client

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def coerce_spec(self, spec_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(spec_data, dict):
            return {"canonical_terms": [], "clinical_scope": ClinicalScope.CLINICIAN_CDS.value}
        data = dict(spec_data)
        data.setdefault("canonical_terms", [])
        if isinstance(data.get("canonical_terms"), str):
            data["canonical_terms"] = [data["canonical_terms"]]
        elif not isinstance(data.get("canonical_terms"), list):
            data["canonical_terms"] = []
        else:
            data["canonical_terms"] = [str(item).strip() for item in data["canonical_terms"] if str(item).strip()]
        specialty = data.get("specialty")
        if isinstance(specialty, list):
            specialty = next((str(item).strip() for item in specialty if str(item).strip()), None)
        elif specialty is not None and not isinstance(specialty, str):
            specialty = str(specialty).strip() or None
        data["specialty"] = specialty or None
        if not isinstance(data.get("patient_context"), dict):
            data["patient_context"] = None
        if not isinstance(data.get("retrieval_hints"), dict):
            data["retrieval_hints"] = {}
        data.setdefault("abbreviations", {})
        data.setdefault("retrieval_hints", {})
        abbreviations = data.get("abbreviations") or {}
        if isinstance(abbreviations, dict):
            cleaned: Dict[str, str] = {}
            for key, value in abbreviations.items():
                if isinstance(value, list):
                    value = next((str(item).strip() for item in value if str(item).strip()), "")
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    cleaned[str(key)] = text
            data["abbreviations"] = cleaned
        else:
            data["abbreviations"] = {}
        try:
            data["clinical_scope"] = ClinicalScope(data.get("clinical_scope")).value
        except (TypeError, ValueError):
            data["clinical_scope"] = ClinicalScope.CLINICIAN_CDS.value
        coerced = []
        for facet in data.get("facets") or []:
            if isinstance(facet, str):
                facet = {"name": facet}
            if not isinstance(facet, dict):
                continue
            item = dict(facet)
            item["name"] = str(item.get("name") or "clinical_evidence")
            item["required"] = bool(item.get("required", True))
            item["threshold"] = self._coerce_float(item.get("threshold"), 0.70)
            item["weight"] = self._coerce_float(item.get("weight"), 1.0)
            keywords = item.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            item["keywords"] = [str(k) for k in keywords if str(k).strip()]
            coerced.append(item)
        data["facets"] = coerced
        return data

    @staticmethod
    def build_facets(
        query: str,
        query_spec: Optional[QuerySpec] = None,
        constraints: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
    ) -> List[ClinicalFacet]:
        constraints = constraints or {}
        explicit = constraints.get("required_facets") or []
        if explicit:
            return [
                ClinicalFacet(**item) if isinstance(item, dict) else ClinicalFacet(name=str(item))
                for item in benchmark_required_facets(None, explicit)
            ]
        threshold = settings.SUFF_CRITICAL_FACET_THRESHOLD
        query_text = (query or "").lower()
        patient_required = bool(patient_id) or bool(re.search(r"\bpatient\b", query_text)) or bool(re.search(r"\belderly\b", query_text))
        fallback = [
            ClinicalFacet(
                name="guideline_concordance",
                required=True,
                threshold=threshold,
                weight=1.20,
                source_policy="CPG",
                keywords=["guideline", "recommendation", "recommended", "indication", "management", "treatment"],
            ),
            ClinicalFacet(
                name="safety_contraindications",
                required=True,
                threshold=threshold,
                weight=1.35,
                source_policy="SAFETY",
                keywords=["contraindication", "contraindicated", "avoid", "adverse", "allergy", "interaction"],
            ),
            ClinicalFacet(
                name="patient_applicability",
                required=patient_required,
                threshold=settings.SUFF_FACET_THRESHOLD,
                weight=1.05,
                source_policy="EMR" if patient_required else "ANY",
                keywords=["patient", "history", "medication", "allergy", "comorbidity"],
            ),
            ClinicalFacet(
                name="evidence_quality",
                required=True,
                threshold=settings.SUFF_FACET_THRESHOLD,
                weight=0.95,
                source_policy="ANY",
                keywords=["grade", "evidence", "trial", "review", "recommendation", "version"],
            ),
        ]
        if not query_spec or not query_spec.facets:
            return fallback
        # Paper: LLM facets enrich F(q); they must not drop the critical fallback set.
        merged = {facet.name: facet for facet in fallback}
        for facet in query_spec.facets:
            current = merged.get(facet.name)
            if current is None:
                merged[facet.name] = facet
                continue
            merged[facet.name] = current.model_copy(
                update={
                    "required": current.required or facet.required,
                    "threshold": max(current.threshold, facet.threshold) if facet.threshold else current.threshold,
                    "weight": max(current.weight, facet.weight),
                    "keywords": list(dict.fromkeys(list(current.keywords) + list(facet.keywords))),
                    "source_policy": facet.source_policy or current.source_policy,
                }
            )
        return list(merged.values())

    async def normalize(self, query: str, trace: Optional[AuditTrace] = None) -> QuerySpec:
        try:
            local_cap = 48 if getattr(self.client, "backend", "") in {"medswin_local", "local_http"} else None
            response = await self.client.call_llm(
                [
                    {"role": "system", "content": query_prompt.SYSTEM},
                    {"role": "user", "content": f"Normalize this medical query: {query}"},
                ],
                json_schema=query_prompt.SCHEMA,
                max_tokens=local_cap,
            )
            spec_data = self.coerce_spec(extract_json_object(response["content"]))
            query_spec = QuerySpec(**spec_data)
            if trace is not None:
                trace.messages.append(
                    AgentMessage(
                        role="assistant",
                        agent_id="normalize",
                        model_endpoint=settings.SUPERVISOR_URL,
                        content=f"Normalized query: {query_spec.canonical_terms}",
                        token_count=response.get("token_count"),
                    )
                )
            return query_spec
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query normalization failed: %s", exc)
            return QuerySpec(canonical_terms=[query])
