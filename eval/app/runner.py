from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from .audit import aggregate_run, audit_case
from .config import Settings
from .io import read_jsonl_cases, write_json
from .client import MedSwinClient
from .schemas import RunAudit, RunRequest


def _gold_doc_ids(cases: list[Any]) -> set[str]:
    doc_ids: set[str] = set()
    for case in cases:
        for doc_id in getattr(case, "gold_doc_ids", []) or []:
            text = str(doc_id).strip()
            if text:
                doc_ids.add(text)
    return doc_ids


def _indexed_doc_ids(stats: dict[str, Any]) -> set[str]:
    manifest = stats.get("index_manifest") or {}
    doc_ids = manifest.get("doc_ids") or []
    if not isinstance(doc_ids, list):
        return set()
    return {str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()}


def _qrel_corpus_diagnostics(cases: list[Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether TREC qrels are actually present in the active corpus.

    Root Cause vs Logic: prior 510-case runs could start with a valid-looking
    index that contained few or no judged PMC documents. The benchmark then
    measured safe abstention instead of end-to-end RAG. This preflight exposes
    the qrel/corpus overlap before the run spends cloud quota.
    """
    gold_doc_ids = _gold_doc_ids(cases)
    stats_gold = set(str(doc_id).strip() for doc_id in (stats.get("gold_doc_ids_present") or stats.get("active_doc_ids") or []) if str(doc_id).strip())
    indexed = _indexed_doc_ids(stats)
    corpus_overlap = gold_doc_ids & stats_gold if stats_gold else set()
    index_overlap = gold_doc_ids & indexed if indexed else set()
    return {
        "gold_doc_count": len(gold_doc_ids),
        "gold_docs_present_in_corpus": len(corpus_overlap) if stats_gold else None,
        "gold_docs_present_in_index": len(index_overlap) if indexed else None,
        "gold_corpus_recall": (len(corpus_overlap) / len(gold_doc_ids)) if gold_doc_ids and stats_gold else None,
        "gold_index_recall": (len(index_overlap) / len(gold_doc_ids)) if gold_doc_ids and indexed else None,
        "missing_gold_doc_ids_sample": sorted(gold_doc_ids - (stats_gold or indexed))[:20],
    }


def _validate_qrel_coverage(diagnostics: dict[str, Any], *, min_corpus_recall: float, min_index_recall: float) -> None:
    gold_count = int(diagnostics.get("gold_doc_count") or 0)
    if gold_count <= 0:
        return
    errors: list[str] = []
    corpus_recall = diagnostics.get("gold_corpus_recall")
    index_recall = diagnostics.get("gold_index_recall")
    if corpus_recall is None:
        errors.append("benchmark runtime did not report active corpus doc ids for qrel coverage")
    elif float(corpus_recall) < min_corpus_recall:
        errors.append(f"gold corpus recall {float(corpus_recall):.3f} is below required {min_corpus_recall:.3f}")
    if index_recall is None:
        errors.append("index manifest did not report doc ids for qrel coverage")
    elif float(index_recall) < min_index_recall:
        errors.append(f"gold index recall {float(index_recall):.3f} is below required {min_index_recall:.3f}")
    if errors:
        sample = diagnostics.get("missing_gold_doc_ids_sample") or []
        raise RuntimeError(
            "Benchmark qrel coverage validation failed: "
            + "; ".join(errors)
            + (f"; missing sample={sample}" if sample else "")
        )


def _validate_setup_stats(stats: dict[str, Any], benchmark_org_id: str) -> None:
    """Fail fast when the benchmark would reuse a stale or foreign index.

    Root Cause vs Logic: a raw `index_exists` check is not enough for a shared
    runtime because the file may belong to a different org, corpus, or embedding
    space. The benchmark must validate provenance before it starts spending
    quota on cases.
    """
    source_counts = stats.get("source_counts") or {}
    manifest = stats.get("index_manifest") or {}
    errors: list[str] = []

    if int(source_counts.get("LIT") or 0) <= 0:
        errors.append("LIT corpus is missing")
    if not bool(stats.get("index_exists")):
        errors.append("active retrieval index file is missing")
    if not manifest:
        errors.append("index provenance manifest is missing")
    if manifest and manifest.get("org_id") != benchmark_org_id:
        errors.append(f"index org_id {manifest.get('org_id')!r} does not match benchmark org {benchmark_org_id!r}")
    if manifest and manifest.get("embedding_space") != stats.get("active_embedding_space"):
        errors.append("index embedding space does not match the active embedding space")
    if manifest and manifest.get("embedding_model") != stats.get("active_embedding_model"):
        errors.append("index embedding model does not match the active embedding model")
    if manifest and int(manifest.get("embedding_dim") or 0) != int(stats.get("active_embedding_dim") or 0):
        errors.append("index embedding dimension does not match the active embedding dimension")
    if manifest and stats.get("active_embeddings") is not None and int(manifest.get("total_vectors") or 0) != int(stats.get("active_embeddings") or 0):
        errors.append("index vector count does not match active benchmark-org embeddings")
    if manifest and "source_counts" in manifest:
        manifest_sources = manifest.get("source_counts") or {}
        if int(manifest_sources.get("LIT") or 0) <= 0 and int(source_counts.get("LIT") or 0) > 0:
            errors.append("index manifest reports no LIT vectors despite benchmark LIT corpus")
    if stats.get("index_provenance_error"):
        errors.append(str(stats["index_provenance_error"]))

    if errors:
        raise RuntimeError("Benchmark setup provenance validation failed: " + "; ".join(errors))


def _aggregate_rate_limit_stats(cases: list[Any]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for case in cases:
        stats = getattr(case, "trace_rate_limit_stats", None) or {}
        for key, payload in stats.items():
            bucket = totals.setdefault(
                key,
                {
                    "rate_limit_events": 0,
                    "retry_events": 0,
                    "cooldown_events": 0,
                },
            )
            bucket["rate_limit_events"] += int((payload or {}).get("rate_limit_events") or 0)
            bucket["retry_events"] += int((payload or {}).get("retry_events") or 0)
            bucket["cooldown_events"] += int(1 if (payload or {}).get("last_delay") else 0)
    return totals


def _case_context_already_materialized(stats: dict[str, Any], benchmark_org_id: str, case_count: int) -> bool:
    """Return True when the benchmark org already has the expected EMR corpus.

    Root Cause vs Logic: the runner was always re-ingesting benchmark notes even
    when the fixed benchmark org already held the matching EMR set and a valid
    provenance-stamped index. That made smoke probes and full audits spend most
    of their wall clock rebuilding state that was already in place. The logic
    below treats the existing benchmark corpus as reusable only when the index
    provenance matches the benchmark org and the EMR chunk count covers the
    current case set.
    """
    if not stats.get("index_exists") or not stats.get("index_provenance_valid"):
        return False
    manifest = stats.get("index_manifest") or {}
    if manifest.get("org_id") != benchmark_org_id:
        return False
    if manifest.get("embedding_space") != stats.get("active_embedding_space"):
        return False
    source_counts = stats.get("source_counts") or {}
    return int(source_counts.get("EMR") or 0) >= int(case_count)


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
    reranker_budget = max(1, min(int(req.reranker_budget or 1), int(req.max_concurrency or 1) if req.max_concurrency else 1))
    case_concurrency = max(1, min(int(req.max_concurrency or 1), reranker_budget))
    setup_concurrency = max(1, min(int(req.max_concurrency or 1), 4))

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
            "reranker_budget": reranker_budget,
            "case_concurrency": case_concurrency,
            "setup_concurrency": setup_concurrency,
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

        setup_stats_before = await client.storage_stats(org_id=benchmark_org_id)
        setup_qrel_before = _qrel_corpus_diagnostics(cases, setup_stats_before)
        if cases:
            _validate_setup_stats(setup_stats_before, benchmark_org_id)
            _validate_qrel_coverage(
                setup_qrel_before,
                min_corpus_recall=settings.benchmark_min_gold_corpus_recall,
                min_index_recall=settings.benchmark_min_gold_index_recall,
            )

        case_context_ready = _case_context_already_materialized(setup_stats_before, benchmark_org_id, len(cases))
        preindexed_context = req.ingest_case_context and hasattr(client, "build_index") and not case_context_ready
        ingest_case_context_per_case = req.ingest_case_context and not case_context_ready and not preindexed_context
        setup_stats_after = setup_stats_before
        if preindexed_context:
            # Motivation vs Logic: benchmark EMR notes must be visible to dense
            # retrieval, not only stored in Mongo after the ANN index is built.
            # Serial note ingestion makes a 510-case rerun spend most of its wall
            # clock on setup instead of evaluation, so we keep the same bounded
            # safety envelope as the benchmark chat phase and ingest notes
            # concurrently before rebuilding the index once.
            ingest_concurrency = setup_concurrency
            ingest_semaphore = asyncio.Semaphore(ingest_concurrency)

            async def bounded_ingest(case: Any) -> None:
                async with ingest_semaphore:
                    await client.ingest_case_context(case)

            await asyncio.gather(*(bounded_ingest(case) for case in cases))
            if hasattr(client, "build_index"):
                await client.build_index(force_rebuild=True, org_id=benchmark_org_id)
            setup_stats_after = await client.storage_stats(org_id=benchmark_org_id)
            _validate_setup_stats(setup_stats_after, benchmark_org_id)
        setup_qrel_after = _qrel_corpus_diagnostics(cases, setup_stats_after)
        if cases:
            _validate_qrel_coverage(
                setup_qrel_after,
                min_corpus_recall=settings.benchmark_min_gold_corpus_recall,
                min_index_recall=settings.benchmark_min_gold_index_recall,
            )
        available_doc_ids = {
            str(doc_id).strip()
            for doc_id in (setup_stats_after.get("active_doc_ids") or [])
            if str(doc_id).strip()
        }
        indexed_doc_ids = _indexed_doc_ids(setup_stats_after)

        async def process_case(case: Any) -> tuple[int, Any]:
            errors: list[str] = []
            response: dict[str, Any] = {}
            trace_summary: dict[str, Any] | None = None
            try:
                if ingest_case_context_per_case:
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
            return 0, audit_case(
                case,
                response,
                trace_summary,
                errors=errors,
                available_doc_ids=available_doc_ids or None,
                indexed_doc_ids=indexed_doc_ids or None,
            )

        if case_concurrency == 1:
            for case in cases:
                _, case_audit = await process_case(case)
                run.cases.append(case_audit)
        else:
            # Motivation vs Logic: a 500-case stress run is only useful if the
            # benchmark can keep the app busy without spending the entire wall
            # clock on strictly serial orchestration. We keep the evaluation
            # semantics intact, but allow bounded concurrency so the harness can
            # exercise the live MedSwin stack at a realistic batch size.
            semaphore = asyncio.Semaphore(case_concurrency)

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
    run.diagnostics = {
        "setup_stats_before": setup_stats_before,
        "setup_stats_after": setup_stats_after,
        "qrel_corpus_before": setup_qrel_before,
        "qrel_corpus_after": setup_qrel_after,
        "case_context_ready": case_context_ready,
        "request_stats": dict(client.request_stats),
        "reranker_budget": reranker_budget,
        "case_concurrency": case_concurrency,
        "setup_concurrency": setup_concurrency,
        "failure_buckets": dict(Counter(case.failure_bucket for case in run.cases if case.failure_bucket)),
        "trace_rate_limit_stats": _aggregate_rate_limit_stats(run.cases),
    }
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
