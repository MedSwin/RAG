"""Aggregate naive-RAG vs full MedSwin audits into one comparison file."""

from __future__ import annotations

from typing import Any

from .schemas import RunAudit


_NUMERIC = [
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
    "error_rate",
]


def compare_audits(naive: RunAudit, medswin: RunAudit) -> dict[str, Any]:
    naive_agg = naive.aggregate or {}
    medswin_agg = medswin.aggregate or {}
    delta = {}
    for key in _NUMERIC:
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
        per_case.append(
            {
                "case_id": case.case_id,
                "naive_msas": baseline.msas,
                "medswin_msas": case.msas,
                "msas_delta": case.msas - baseline.msas,
                "naive_evidence_doc_recall": baseline.evidence_doc_recall,
                "medswin_evidence_doc_recall": case.evidence_doc_recall,
                "recall_delta": case.evidence_doc_recall - baseline.evidence_doc_recall,
                "naive_policy_passed": baseline.policy_passed,
                "medswin_policy_passed": case.policy_passed,
                "medswin_abstained": case.policy_passed is False,
                "jaccard": (len(overlap) / len(union)) if union else 0.0,
                "overlap_chunk_ids": sorted(overlap),
            }
        )

    return {
        "naive_run_id": naive.run_id,
        "medswin_run_id": medswin.run_id,
        "naive_aggregate": naive_agg,
        "medswin_aggregate": medswin_agg,
        "delta_medswin_minus_naive": delta,
        "per_case": per_case,
    }
