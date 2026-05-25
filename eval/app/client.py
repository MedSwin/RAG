from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from facets import benchmark_required_facets
from .schemas import BenchmarkCase


def _truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text or "").split()[:max_words]).strip()


class MedSwinClient:
    def __init__(
        self,
        base_url: str,
        org_id: str,
        user_id: str,
        timeout_s: float = 120.0,
        include_patient_context_in_query: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.user_id = user_id
        self.timeout_s = timeout_s
        self.include_patient_context_in_query = include_patient_context_in_query
        self._client: httpx.AsyncClient | None = None
        self.request_stats: dict[str, int] = {
            "retries": 0,
            "rate_limits": 0,
            "timeouts": 0,
            "network_errors": 0,
        }

    async def __aenter__(self) -> "MedSwinClient":
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("MedSwinClient must be used as an async context manager")
        return self._client

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/health")

    async def ingest_case_context(self, case: BenchmarkCase) -> dict[str, Any] | None:
        """Ingest the TREC topic note as EMR-like context scoped to the case patient_id.

        The literature corpus should be bulk-ingested separately from TREC/PMC.
        This per-case ingest only registers the patient-specific note so that the
        system can test patient-context alignment and PHI-safe trace handling.
        """
        if not case.patient_context:
            return None
        patient_id = case.patient_id or f"patient-{case.case_id}"
        note_excerpt = _truncate_words(case.patient_context, 900)
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
                # Root Cause vs Logic: raw TREC admission notes can exceed the
                # evidence token budget as a single chunk, making patient
                # applicability unselectable even after successful EMR
                # retrieval. The benchmark keeps the full note on the document
                # but materializes a compact retrieval chunk that can fit beside
                # literature evidence.
                "chunks": [
                    {
                        "chunk_id": f"{case.dataset}:{case.case_id}:note_chunk_0",
                        "text": note_excerpt,
                        "section": "patient_context_excerpt",
                        "offset_start": 0,
                        "offset_end": len(note_excerpt),
                        "metadata": {"benchmark_patient_context_excerpt": True},
                    }
                ],
            }
        ]
        return await self._request_json(
            "POST",
            "/api/v1/medswin/ingest",
            params={"source_type": "EMR", "org_id": self.org_id},
            json=payload,
        )

    async def chat(self, case: BenchmarkCase, *, source_policy: str, guideline_only: bool, min_evidence_grade: float, clinical_scope: str) -> dict[str, Any]:
        patient_id = case.patient_id or f"patient-{case.case_id}"
        required_facets = benchmark_required_facets(case.query_type, case.gold_facets)

        constraints = {
            "clinical_scope": clinical_scope,
            "guideline_only": guideline_only,
            "required_facets": [facet for facet in required_facets if facet.get("critical")],
            "source_policy": source_policy,
            "min_evidence_grade": min_evidence_grade,
            **case.constraints,
        }
        full_query = case.query
        if self.include_patient_context_in_query and case.patient_context:
            # Root Cause vs Logic: the previous harness concatenated patient_context
            # into the benchmark query, which measured retrieval plus prompt augmentation.
            # The logic now keeps the query pure and relies on explicit EMR ingestion so
            # the benchmark reflects the runtime contract.
            full_query = f"Patient context:\n{case.patient_context}\n\nClinical question:\n{case.query}"
        payload = {
            "query": full_query,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "patient_id": patient_id,
            "constraints": constraints,
        }
        return await self._request_json("POST", "/api/v1/medswin/chat", json=payload)

    async def build_index(self, *, force_rebuild: bool = True, org_id: str | None = None) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/api/v1/storage/index/build",
            json={"force_rebuild": force_rebuild, "org_id": org_id},
        )

    async def reset_benchmark_org(self, *, org_id: str, remove_indexes: bool = True) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/api/v1/storage/benchmark/reset",
            json={"org_id": org_id, "remove_indexes": remove_indexes},
        )

    async def trace(self, trace_id: str, include_details: bool = True) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/api/v1/medswin/traces/{trace_id}",
            params={"org_id": self.org_id, "include_details": str(include_details).lower()},
        )

    async def storage_stats(self, org_id: str | None = None) -> dict[str, Any]:
        params = {"org_id": org_id} if org_id else None
        return await self._request_json("GET", "/api/v1/storage/stats", params=params)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        """Retry transient benchmark setup and chat calls.

        Root Cause vs Logic: a large benchmark can hit transient 429/5xx bursts
        while the runtime warms indexes, builds embeddings, or backs off a
        shared model. We retry the HTTP edge here so the eval pipeline measures
        the system rather than a single transient request failure.
        """
        last_exc: Exception | None = None
        url = f"{self.base_url}{path}"
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self.client.request(method, url, params=params, json=json)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    self.request_stats["rate_limits" if resp.status_code == 429 else "retries"] += 1
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    if attempt == max_attempts:
                        resp.raise_for_status()
                    await asyncio.sleep(self._retry_delay(attempt, resp.headers.get("Retry-After")))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.TimeoutException):
                    self.request_stats["timeouts"] += 1
                else:
                    self.request_stats["network_errors"] += 1
                if attempt == max_attempts:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempt == max_attempts:
                    raise
                await asyncio.sleep(self._retry_delay(attempt, exc.response.headers.get("Retry-After")))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Request failed for {path}")

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        base_delay = min(30.0, 1.5 * (2 ** (attempt - 1)))
        if retry_after:
            try:
                hinted = float(retry_after)
            except ValueError:
                hinted = base_delay
            else:
                hinted = max(0.0, hinted)
            return max(base_delay, hinted) + random.uniform(0.0, 0.25)
        return base_delay + random.uniform(0.0, 0.25)
