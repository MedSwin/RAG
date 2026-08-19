"""Aggregate naive-RAG vs full MedSwin audits into one comparison file."""

from __future__ import annotations

from typing import Any

from .schemas import RunAudit


_NUMERIC = [
    # Pre-pack retrieval metrics are populated by the strict full matrix.
    "mean_retrieval_ndcg_at_10",
    "mean_retrieval_precision_at_10",
    "mean_retrieval_recall_at_10",
    "mean_retrieval_reciprocal_rank",
    # Final RAG evidence packet under the shared context budget.
    "mean_final_evidence_ndcg_at_10",
    "mean_final_evidence_precision_at_10",
    "mean_final_evidence_recall_at_10",
    "mean_final_evidence_reciprocal_rank",
    # System-audit diagnostics.
    "mean_facet_recall",
    "mean_critical_facet_recall",
    "mean_evidence_doc_recall",
    "mean_citation_precision",
    "mean_trace_completeness",
    "mean_sufficiency_decision_score",
    "mean_groundedness_proxy",
    "mean_unsupported_penalty",
    "mean_unsafe_omission_penalty",
    "mean_msas",
    "policy_pass_rate",
    "degraded_rate",
    "error_rate",
]


def compare_audits(naive: RunAudit, medswin: RunAudit) -> dict[str, Any]:
    naive_agg = naive.aggregate or {}
    medswin_agg = medswin.aggregate or {}
    delta = {}
    for key in _NUMERIC:
        # Do not invent a zero for an optional retrieval metric that was not
        # measured by a generic/smoke audit. Emit None unless both runs contain
        # the metric so callers can distinguish "not measured" from zero score.
        if key.startswith("mean_retrieval_") and (key not in naive_agg or key not in medswin_agg):
            delta[key] = None
            continue
        left = float(medswin_agg.get(key) or 0.0)
        right = float(naive_agg.get(key) or 0.0)
        delta[key] = left - right

    naive_cases = {case.case_id: case for case in naive.cases}
    per_case = []
    for case in medswin.cases:
        baseline = naive_cases.get(case.case_id)
        if baseline is None:
            continue
        naive_ids = set(baseline.selected_chunk_ids)
        medswin_ids = set(case.selected_chunk_ids)
        overlap = naive_ids & medswin_ids
        union = naive_ids | medswin_ids
        retrieval_delta = None
        if baseline.retrieval_ndcg_at_10 is not None and case.retrieval_ndcg_at_10 is not None:
            retrieval_delta = case.retrieval_ndcg_at_10 - baseline.retrieval_ndcg_at_10
        per_case.append(
            {
                "case_id": case.case_id,
                "naive_retrieval_ndcg_at_10": baseline.retrieval_ndcg_at_10,
                "medswin_retrieval_ndcg_at_10": case.retrieval_ndcg_at_10,
                "retrieval_ndcg_at_10_delta": retrieval_delta,
                "naive_final_evidence_ndcg_at_10": baseline.final_evidence_ndcg_at_10,
                "medswin_final_evidence_ndcg_at_10": case.final_evidence_ndcg_at_10,
                "final_evidence_ndcg_at_10_delta": (
                    case.final_evidence_ndcg_at_10 - baseline.final_evidence_ndcg_at_10
                ),
                "naive_msas": baseline.msas,
                "medswin_msas": case.msas,
                "msas_delta": case.msas - baseline.msas,
                "naive_evidence_doc_recall": baseline.evidence_doc_recall,
                "medswin_evidence_doc_recall": case.evidence_doc_recall,
                "recall_delta": case.evidence_doc_recall - baseline.evidence_doc_recall,
                "naive_policy_passed": baseline.policy_passed,
                "medswin_policy_passed": case.policy_passed,
                "medswin_abstained": case.policy_passed is False,
                "naive_retrieval_backend": baseline.retrieval_backend,
                "medswin_retrieval_backend": case.retrieval_backend,
                "naive_errors": list(baseline.errors),
                "medswin_errors": list(case.errors),
                "jaccard": (len(overlap) / len(union)) if union else 0.0,
                "overlap_chunk_ids": sorted(overlap),
            }
        )

    config_keys = (
        "pipeline",
        "top_k",
        "retrieval_top_k",
        "min_evidence_grade",
        "source_policy",
        "guideline_only",
        "case_concurrency",
        "reranker_budget",
        "benchmark_org_id",
        "shared_generation_envelope",
    )
    return {
        "naive_run_id": naive.run_id,
        "medswin_run_id": medswin.run_id,
        "naive_aggregate": naive_agg,
        "medswin_aggregate": medswin_agg,
        "delta_medswin_minus_naive": delta,
        "naive_retrieval_backend_counts": (naive.diagnostics or {}).get("retrieval_backend_counts") or {},
        "medswin_retrieval_backend_counts": (medswin.diagnostics or {}).get("retrieval_backend_counts") or {},
        "naive_config": {key: (naive.config or {}).get(key) for key in config_keys},
        "medswin_config": {key: (medswin.config or {}).get(key) for key in config_keys},
        "per_case": per_case,
    }
