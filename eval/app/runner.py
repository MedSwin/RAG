from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from .audit import aggregate_run, audit_case
from .config import Settings
from .io import read_jsonl_cases, write_json
from .client import MedSwinClient
from .schemas import RunAudit, RunRequest


async def run_benchmark(req: RunRequest, settings: Settings) -> RunAudit:
    cases = read_jsonl_cases(req.cases_path)
    if req.max_cases is not None:
        cases = cases[: req.max_cases]
    elif settings.max_cases_default:
        cases = cases[: settings.max_cases_default]

    run_id = str(uuid.uuid4())
    # Motivation vs Logic: benchmark runs need tenant isolation so repeated audits
    # do not collide through shared document and patient namespaces. Scoping each
    # run to a unique org_id keeps the corpus deterministic and reproducible.
    benchmark_org_id = f"{settings.benchmark_org_id}-{run_id}"

    run = RunAudit(
        run_id=run_id,
        dataset=cases[0].dataset if cases else "unknown",
        config={
            "cases_path": req.cases_path,
            "max_cases": len(cases),
            "medswin_base_url": settings.medswin_base_url,
            "benchmark_org_id": benchmark_org_id,
            "source_policy": req.source_policy,
            "guideline_only": req.guideline_only,
            "min_evidence_grade": req.min_evidence_grade,
            "clinical_scope": req.clinical_scope,
            "ingest_case_context": req.ingest_case_context,
            "include_patient_context_in_query": req.include_patient_context_in_query,
        },
    )

    async with MedSwinClient(
        base_url=settings.medswin_base_url,
        org_id=benchmark_org_id,
        user_id=settings.benchmark_user_id,
        timeout_s=settings.request_timeout_s,
        include_patient_context_in_query=req.include_patient_context_in_query,
    ) as client:
        try:
            await client.health()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"MedSwin runtime health check failed: {exc}") from exc

        for case in cases:
            errors: list[str] = []
            response: dict[str, Any] = {}
            trace_summary: dict[str, Any] | None = None
            try:
                if req.ingest_case_context:
                    await client.ingest_case_context(case)
                response = await client.chat(
                    case,
                    source_policy=req.source_policy,
                    guideline_only=req.guideline_only,
                    min_evidence_grade=req.min_evidence_grade,
                    clinical_scope=req.clinical_scope,
                )
                trace_id = response.get("trace_id") or (response.get("trace") or {}).get("trace_id")
                if trace_id:
                    try:
                        trace_summary = await client.trace(trace_id, include_details=True)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"trace_fetch_failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"case_failed: {exc}")
                response = {"answer": "", "policy_decision": {"passed": False}, "citations": [], "evidence_bundle": {}}
            run.cases.append(audit_case(case, response, trace_summary, errors=errors))

    aggregate_run(run)
    output_path = Path(settings.run_store_dir) / f"{run_id}.json"
    write_json(output_path, run.model_dump())
    return run


def run_benchmark_sync(req: RunRequest, settings: Settings) -> RunAudit:
    return asyncio.run(run_benchmark(req, settings))
