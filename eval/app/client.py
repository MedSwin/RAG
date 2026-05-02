from __future__ import annotations

from typing import Any
import httpx

from .schemas import BenchmarkCase


class MedSwinClient:
    def __init__(self, base_url: str, org_id: str, user_id: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.user_id = user_id
        self.timeout_s = timeout_s

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def ingest_case_context(self, case: BenchmarkCase) -> dict[str, Any] | None:
        """Ingest the TREC topic note as EMR-like context scoped to the case patient_id.

        The literature corpus should be bulk-ingested separately from TREC/PMC.
        This per-case ingest only registers the patient-specific note so that the
        system can test patient-context alignment and PHI-safe trace handling.
        """
        if not case.patient_context:
            return None
        patient_id = case.patient_id or f"patient-{case.case_id}"
        payload = [
            {
                "doc_id": f"{case.dataset}:{case.case_id}:note",
                "title": f"TREC CDS case note {case.case_id}",
                "version": "benchmark",
                "patient_id": patient_id,
                "source_reliability": 0.8,
                "evidence_grade": {
                    "label": "emr_note",
                    "score": 0.7,
                    "source_reliability": 0.8,
                },
                "tags": ["benchmark", case.dataset, case.query_type or "clinical"],
                "metadata": {
                    "benchmark_case_id": case.case_id,
                    "dataset": case.dataset,
                    "query_type": case.query_type,
                },
                "text": case.patient_context,
            }
        ]
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/medswin/ingest",
                params={"source_type": "EMR", "org_id": self.org_id},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def chat(self, case: BenchmarkCase, *, source_policy: str, guideline_only: bool, min_evidence_grade: float, clinical_scope: str) -> dict[str, Any]:
        patient_id = case.patient_id or f"patient-{case.case_id}"
        constraints = {
            "clinical_scope": clinical_scope,
            "guideline_only": guideline_only,
            "required_facets": [f.name for f in case.gold_facets if f.critical],
            "source_policy": source_policy,
            "min_evidence_grade": min_evidence_grade,
            **case.constraints,
        }
        # Include case context in the query. If you prefer strict EMR-only context,
        # remove this prefix after confirming patient-scoped retrieval is reliable.
        full_query = case.query
        if case.patient_context:
            full_query = f"Patient context:\n{case.patient_context}\n\nClinical question:\n{case.query}"
        payload = {
            "query": full_query,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "patient_id": patient_id,
            "constraints": constraints,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/v1/medswin/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def trace(self, trace_id: str, include_details: bool = True) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(
                f"{self.base_url}/api/v1/medswin/traces/{trace_id}",
                params={"org_id": self.org_id, "include_details": str(include_details).lower()},
            )
            resp.raise_for_status()
            return resp.json()
