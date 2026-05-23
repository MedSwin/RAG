from __future__ import annotations

import asyncio
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
    elif settings.benchmark_max_topics:
        cases = cases[: settings.benchmark_max_topics]
    elif settings.max_cases_default:
        cases = cases[: settings.max_cases_default]

    import uuid
    run_id = str(uuid.uuid4())
    # Motivation vs Logic: the benchmark corpus is prepared once in a fixed org
    # namespace, while each case note is still patient-scoped. Using the same org
    # for eval runs makes the 5000-sample corpus visible to /medswin/chat.
    benchmark_org_id = settings.benchmark_org_id

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
            "max_concurrency": req.max_concurrency,
            "ingest_case_context": req.ingest_case_context,
            "fetch_trace_summary": req.fetch_trace_summary,
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

        preindexed_context = req.ingest_case_context and hasattr(client, "build_index")
        if preindexed_context:
            # Motivation vs Logic: benchmark EMR notes must be visible to dense
            # retrieval, not only stored in Mongo after the ANN index is built.
            # Ingest all notes first, then rebuild once so chat exercises the
            # full dense+lexical+rereank MedSwin path.
            for case in cases:
                await client.ingest_case_context(case)
            if hasattr(client, "build_index"):
                await client.build_index(force_rebuild=True)

        async def process_case(case: Any) -> tuple[int, Any]:
            errors: list[str] = []
            response: dict[str, Any] = {}
            trace_summary: dict[str, Any] | None = None
            try:
                if req.ingest_case_context and not preindexed_context:
                    await client.ingest_case_context(case)
                response = await client.chat(
                    case,
                    source_policy=req.source_policy,
                    guideline_only=req.guideline_only,
                    min_evidence_grade=req.min_evidence_grade,
                    clinical_scope=req.clinical_scope,
                )
                trace_id = response.get("trace_id") or (response.get("trace") or {}).get("trace_id")
                if trace_id and req.fetch_trace_summary:
                    try:
                        trace_summary = await client.trace(trace_id, include_details=True)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"trace_fetch_failed: {exc}")
                errors.extend(_architecture_errors(response, trace_summary))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"case_failed: {exc}")
                response = {"answer": "", "policy_decision": {"passed": False}, "citations": [], "evidence_bundle": {}}
            return 0, audit_case(case, response, trace_summary, errors=errors)

        max_concurrency = max(int(req.max_concurrency or 1), 1)
        if max_concurrency == 1:
            for case in cases:
                _, case_audit = await process_case(case)
                run.cases.append(case_audit)
        else:
            # Motivation vs Logic: a 500-case stress run is only useful if the
            # benchmark can keep the app busy without spending the entire wall
            # clock on strictly serial orchestration. We keep the evaluation
            # semantics intact, but allow bounded concurrency so the harness can
            # exercise the live MedSwin stack at a realistic batch size.
            semaphore = asyncio.Semaphore(max_concurrency)

            async def bounded_process(index: int, case: Any) -> tuple[int, Any]:
                async with semaphore:
                    _, case_audit = await process_case(case)
                    return index, case_audit

            tasks = [asyncio.create_task(bounded_process(index, case)) for index, case in enumerate(cases)]
            audits: list[Any | None] = [None] * len(tasks)
            for task in asyncio.as_completed(tasks):
                index, case_audit = await task
                audits[index] = case_audit
            run.cases.extend(case_audit for case_audit in audits if case_audit is not None)

    aggregate_run(run)
    output_path = Path(settings.run_store_dir) / f"{run_id}.json"
    write_json(output_path, run.model_dump())
    return run


def run_benchmark_sync(req: RunRequest, settings: Settings) -> RunAudit:
    return asyncio.run(run_benchmark(req, settings))


def _architecture_errors(response: dict[str, Any], trace_summary: dict[str, Any] | None) -> list[str]:
    """Return errors when a case skips required MedSwin architecture artifacts."""
    errors: list[str] = []
    degraded = response.get("degraded_mode")
    if degraded is True or (isinstance(degraded, dict) and any(degraded.values())):
        errors.append("architecture_degraded")
    evidence_bundle = response.get("evidence_bundle") or {}
    if not evidence_bundle:
        errors.append("missing_evidence_bundle")
    if response.get("policy_decision") is None:
        errors.append("missing_policy_decision")
    if not response.get("trace_id"):
        errors.append("missing_trace_id")
    if trace_summary is None:
        errors.append("missing_trace_summary")
    else:
        if int(trace_summary.get("sufficiency_checks_count") or 0) <= 0:
            errors.append("missing_sufficiency_checks")
        if int(trace_summary.get("messages_count") or 0) <= 0:
            errors.append("missing_agent_messages")
        if int(trace_summary.get("evidence_passages_count") or 0) <= 0:
            errors.append("missing_trace_evidence")
    return errors
