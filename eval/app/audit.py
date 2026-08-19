from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .schemas import BenchmarkCase, CaseAudit, GoldFacet, RunAudit


TREC_EVIDENCE_CUTOFF = 10


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "allow_generation", "generation_allowed", "sufficient"):
            if key in value and isinstance(value[key], bool):
                return value[key]
    return None


def _norm_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _trace_count(trace_summary: dict[str, Any] | None, *keys: str) -> bool:
    if not trace_summary:
        return False
    return any(bool(trace_summary.get(key)) for key in keys)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def extract_ranked_doc_ids(response: dict[str, Any]) -> list[str]:
    """Preserve final literature-evidence order presented to the generator.

    TREC CDS evaluates biomedical article retrieval. Patient EMR context is a
    MedSwin system input, not a TREC result document, so known non-LIT sources do
    not consume ranks. Older fixtures that omit source_type are retained as
    literature-compatible rather than silently returning an empty ranking.
    """
    bundle = response.get("evidence_bundle") or {}
    known_non_lit = {"EMR", "CPG", "SAFETY"}
    for key in ("passages", "evidence", "selected_passages", "chunks", "items"):
        items = bundle.get(key)
        if not isinstance(items, list):
            continue
        ranked: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_type") or item.get("source") or "").upper()
            if source in known_non_lit:
                continue
            doc_id = _norm_id(item.get("doc_id") or item.get("document_id") or item.get("source_id"))
            if doc_id:
                ranked.append(doc_id)
        if ranked:
            return _ordered_unique(ranked)
    # A response should normally expose evidence_bundle.passages. Citations are
    # a last-resort ranked fallback for older audit fixtures.
    citations = response.get("citations") or []
    return _ordered_unique(
        [
            doc_id
            for item in citations
            if isinstance(item, dict)
            for doc_id in [_norm_id(item.get("doc_id") or item.get("document_id") or item.get("source_id"))]
            if doc_id
        ]
    )


def _relevance_grades(case: BenchmarkCase) -> dict[str, int]:
    raw = (case.metadata or {}).get("relevance_grades") or {}
    grades: dict[str, int] = {}
    if isinstance(raw, dict):
        for doc_id, value in raw.items():
            try:
                grade = int(value)
            except (TypeError, ValueError):
                continue
            if grade > 0:
                grades[str(doc_id)] = grade
    # Case-level positive qrels are authoritative if grade metadata is missing.
    for doc_id in case.gold_doc_ids:
        grades.setdefault(str(doc_id), 1)
    return grades


def ranked_trec_metrics(
    case: BenchmarkCase,
    ranked_doc_ids: list[str],
    cutoff: int = TREC_EVIDENCE_CUTOFF,
) -> dict[str, float]:
    """Compute graded nDCG@k plus early binary relevance measures from qrels.

    These are qrel-grounded ranked-evidence diagnostics. They are deliberately
    reported separately from MSAS because MSAS also measures MedSwin-specific
    audit/gating behavior that a naive-RAG control is designed not to implement.
    """
    cutoff = max(1, int(cutoff))
    grades = _relevance_grades(case)
    positive = {doc_id for doc_id, grade in grades.items() if grade > 0}
    ranking = _ordered_unique(ranked_doc_ids)[:cutoff]

    def gain(grade: int) -> float:
        return float((2 ** max(0, int(grade))) - 1)

    dcg = 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        rel = int(grades.get(doc_id, 0))
        if rel > 0:
            dcg += gain(rel) / math.log2(rank + 1)
    ideal = sorted((int(value) for value in grades.values() if int(value) > 0), reverse=True)[:cutoff]
    idcg = sum(gain(rel) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    ndcg = dcg / idcg if idcg > 0 else 1.0

    relevant_retrieved = sum(1 for doc_id in ranking if doc_id in positive)
    precision = relevant_retrieved / float(cutoff)
    recall = relevant_retrieved / len(positive) if positive else 1.0
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in positive:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "ndcg_at_10": max(0.0, min(1.0, ndcg)),
        "precision_at_10": max(0.0, min(1.0, precision)),
        "recall_at_10": max(0.0, min(1.0, recall)),
        "reciprocal_rank": max(0.0, min(1.0, reciprocal_rank)),
    }


def extract_doc_ids(response: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    selected_doc_ids: list[str] = []
    cited_doc_ids: list[str] = []
    selected_chunk_ids: list[str] = []

    bundle = response.get("evidence_bundle") or {}
    citations = response.get("citations") or []
    ledger = response.get("evidence_ledger") or bundle.get("evidence_ledger") or []

    possible_evidence_lists = []
    for key in ("passages", "evidence", "selected_passages", "chunks", "items"):
        if isinstance(bundle.get(key), list):
            possible_evidence_lists.append(bundle[key])
    if isinstance(ledger, list):
        possible_evidence_lists.append(ledger)

    for items in possible_evidence_lists:
        for item in items:
            if not isinstance(item, dict):
                continue
            doc_id = _norm_id(item.get("doc_id") or item.get("document_id") or item.get("source_id"))
            chunk_id = _norm_id(item.get("chunk_id") or item.get("id"))
            if doc_id:
                selected_doc_ids.append(doc_id)
            if chunk_id:
                selected_chunk_ids.append(chunk_id)

    for cit in citations:
        if not isinstance(cit, dict):
            continue
        doc_id = _norm_id(cit.get("doc_id") or cit.get("document_id") or cit.get("source_id"))
        if doc_id:
            cited_doc_ids.append(doc_id)
        chunk_id = _norm_id(cit.get("chunk_id") or cit.get("id"))
        if chunk_id:
            selected_chunk_ids.append(chunk_id)

    return sorted(set(selected_doc_ids)), sorted(set(cited_doc_ids)), sorted(set(selected_chunk_ids))


def selected_source_counts(response: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {"CPG": 0, "EMR": 0, "LIT": 0, "SAFETY": 0}
    bundle = response.get("evidence_bundle") or {}
    for key in ("passages", "evidence", "selected_passages", "chunks", "items"):
        items = bundle.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_type") or item.get("source") or "").upper()
            if source in counts:
                counts[source] += 1
        if sum(counts.values()) > 0:
            break
    return counts


def _failure_bucket(
    *,
    selected_counts: dict[str, int],
    gold_available_in_corpus: int | None,
    gold_available_in_index: int | None,
    evidence_doc_recall: float,
    critical_facet_recall: float,
    policy_passed: bool | None,
    errors: list[str],
) -> str | None:
    if errors:
        return "trace_or_runtime_failure"
    if (gold_available_in_corpus or 0) == 0:
        return "qrel_doc_absent_from_corpus"
    if gold_available_in_index is not None and gold_available_in_index == 0:
        return "qrel_doc_absent_from_index"
    if selected_counts.get("LIT", 0) == 0:
        return "no_literature_retrieved"
    if evidence_doc_recall <= 0.0:
        return "literature_retrieved_but_no_qrel_overlap"
    if critical_facet_recall <= 0.0:
        return "selected_evidence_not_facet_supported"
    if policy_passed is False:
        return "policy_threshold_failure"
    return None


def facet_recall(
    gold_facets: list[GoldFacet],
    evidence_doc_ids: set[str],
    *,
    critical_only: bool = False,
    fallback_gold_doc_ids: set[str] | None = None,
) -> float:
    facets = [f for f in gold_facets if (f.critical or not critical_only)]
    if critical_only:
        facets = [f for f in gold_facets if f.critical]
    if not facets:
        return 1.0
    earned = 0.0
    total = 0.0
    for facet in facets:
        weight = max(0.0, facet.weight)
        total += weight
        gold = set(facet.gold_doc_ids)
        if fallback_gold_doc_ids and (not gold or gold.isdisjoint(fallback_gold_doc_ids)):
            gold = fallback_gold_doc_ids
        if not gold:
            earned += weight
        elif gold & evidence_doc_ids:
            earned += weight
    return earned / total if total else 1.0


def trace_completeness(response: dict[str, Any], trace_summary: dict[str, Any] | None = None) -> float:
    checks = {
        "answer": bool(response.get("answer")),
        "evidence_bundle": response.get("evidence_bundle") is not None,
        "policy_decision": response.get("policy_decision") is not None,
        "facet_coverage": response.get("facet_coverage") is not None,
        "citations": isinstance(response.get("citations"), list),
        "evidence_ledger": (
            response.get("evidence_ledger") is not None
            or (response.get("evidence_bundle") or {}).get("evidence_ledger") is not None
        ),
        "trace_id": bool(response.get("trace_id") or response.get("trace", {}).get("trace_id")),
    }
    if trace_summary is not None:
        checks["trace_fetch"] = bool(trace_summary)
        checks["trace_counts"] = (
            _trace_count(trace_summary, "messages_count", "message_count")
            or _trace_count(trace_summary, "tool_calls_count", "tool_count")
            or _trace_count(trace_summary, "sufficiency_checks_count", "sufficiency_check_count")
            or _trace_count(trace_summary, "evidence_passages_count", "evidence_count")
        )
    return sum(checks.values()) / len(checks)


def groundedness_proxy(response: dict[str, Any], cited_doc_ids: set[str]) -> tuple[float, float]:
    """Estimate citation/ledger alignment without an external semantic judge.

    This remains an automatic system diagnostic. It is not presented as a
    replacement for clinician/rubric-based claim adjudication.
    """
    answer = response.get("answer") or ""
    ledger = response.get("evidence_ledger") or (response.get("evidence_bundle") or {}).get("evidence_ledger") or []
    citations = response.get("citations") or []
    if not answer:
        return 0.0, 1.0
    if isinstance(ledger, list) and ledger:
        supported = 0
        total = 0
        for item in ledger:
            if not isinstance(item, dict):
                continue
            for claim_item in _ledger_claim_items(item):
                total += 1
                doc_id = _norm_id(claim_item.get("doc_id"))
                polarity = str(claim_item.get("polarity") or "support").lower()
                if doc_id in cited_doc_ids and polarity in {"support", "supports", "qualifies"}:
                    supported += 1
        if total:
            score = supported / total
            return score, 1.0 - score
    if citations:
        return 0.65, 0.35
    return 0.25, 0.75


def _ledger_claim_items(item: dict[str, Any]) -> list[dict[str, Any]]:
    entry_doc_id = item.get("doc_id") or item.get("document_id") or item.get("source_id")
    top_claim = item.get("claim") or item.get("text") or item.get("statement")
    if top_claim:
        return [
            {
                "claim": top_claim,
                "doc_id": entry_doc_id,
                "polarity": item.get("polarity", "support"),
            }
        ]
    nested = item.get("claims") if isinstance(item.get("claims"), list) else []
    claims: list[dict[str, Any]] = []
    for sub in nested:
        if not isinstance(sub, dict):
            continue
        claim = sub.get("claim") or sub.get("text") or sub.get("statement")
        if not claim:
            continue
        claims.append(
            {
                "claim": claim,
                "doc_id": sub.get("doc_id") or sub.get("document_id") or entry_doc_id,
                "polarity": sub.get("polarity") or item.get("polarity") or "support",
            }
        )
    return claims


def audit_case(
    case: BenchmarkCase,
    response: dict[str, Any],
    trace_summary: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    available_doc_ids: set[str] | None = None,
    indexed_doc_ids: set[str] | None = None,
    pipeline: str | None = None,
) -> CaseAudit:
    ranked_doc_ids = extract_ranked_doc_ids(response)
    trec_metrics = ranked_trec_metrics(case, ranked_doc_ids)
    selected_doc_ids, cited_doc_ids, selected_chunk_ids = extract_doc_ids(response)
    selected_counts = selected_source_counts(response)
    evidence_doc_ids = set(selected_doc_ids) | set(cited_doc_ids)
    gold_doc_ids = set(case.gold_doc_ids)
    available_gold = len(gold_doc_ids & available_doc_ids) if available_doc_ids is not None else None
    indexed_gold = len(gold_doc_ids & indexed_doc_ids) if indexed_doc_ids is not None else None

    evidence_doc_recall = len(gold_doc_ids & evidence_doc_ids) / len(gold_doc_ids) if gold_doc_ids else 1.0
    citation_precision = (
        len(set(cited_doc_ids) & gold_doc_ids) / len(set(cited_doc_ids))
        if cited_doc_ids and gold_doc_ids
        else (1.0 if cited_doc_ids else 0.0)
    )

    f_recall = facet_recall(case.gold_facets, evidence_doc_ids, critical_only=False, fallback_gold_doc_ids=gold_doc_ids)
    cf_recall = facet_recall(case.gold_facets, evidence_doc_ids, critical_only=True, fallback_gold_doc_ids=gold_doc_ids)

    policy_decision = response.get("policy_decision") or {}
    policy_passed = _as_bool(policy_decision)
    has_critical_gap = cf_recall < 1.0
    if policy_passed is None:
        suff_score = 0.5
    elif policy_passed and has_critical_gap:
        suff_score = 0.0
    elif not policy_passed and not has_critical_gap:
        suff_score = 0.5
    else:
        suff_score = 1.0

    ground_score, unsupported_penalty = groundedness_proxy(response, set(cited_doc_ids))
    unsafe_penalty = max(0.0, 1.0 - cf_recall)
    trace_score = trace_completeness(response, trace_summary)

    # MSAS stays a system-audit diagnostic. The previous formula rewarded raw
    # answer length as a "clinical quality" proxy; verbosity is not clinical
    # quality, so that term is replaced by qrel-grounded citation precision.
    msas = (
        0.25 * cf_recall
        + 0.15 * f_recall
        + 0.15 * ground_score
        + 0.15 * suff_score
        + 0.10 * trace_score
        + 0.10 * evidence_doc_recall
        + 0.10 * citation_precision
        - 0.20 * unsafe_penalty
        - 0.10 * unsupported_penalty
    )
    msas = max(0.0, min(1.0, msas))

    return CaseAudit(
        case_id=case.case_id,
        dataset=case.dataset,
        pipeline=pipeline or response.get("pipeline"),
        retrieval_backend=response.get("retrieval_backend"),
        trace_id=response.get("trace_id") or (response.get("trace") or {}).get("trace_id"),
        session_id=response.get("session_id"),
        policy_passed=policy_passed,
        degraded_mode=bool(response.get("degraded_mode")) if response.get("degraded_mode") is not None else None,
        answer_chars=len(str(response.get("answer") or "")),
        ranked_doc_ids=ranked_doc_ids,
        selected_doc_ids=selected_doc_ids,
        cited_doc_ids=cited_doc_ids,
        selected_chunk_ids=selected_chunk_ids,
        selected_source_counts=selected_counts,
        gold_doc_ids=sorted(gold_doc_ids),
        gold_available_in_corpus=available_gold,
        gold_available_in_index=indexed_gold,
        gold_available_but_not_retrieved=(
            available_gold is not None and available_gold > 0 and evidence_doc_recall <= 0.0
        ),
        failure_bucket=_failure_bucket(
            selected_counts=selected_counts,
            gold_available_in_corpus=available_gold,
            gold_available_in_index=indexed_gold,
            evidence_doc_recall=evidence_doc_recall,
            critical_facet_recall=cf_recall,
            policy_passed=policy_passed,
            errors=errors or [],
        ),
        ndcg_at_10=trec_metrics["ndcg_at_10"],
        precision_at_10=trec_metrics["precision_at_10"],
        recall_at_10=trec_metrics["recall_at_10"],
        reciprocal_rank=trec_metrics["reciprocal_rank"],
        facet_recall=f_recall,
        critical_facet_recall=cf_recall,
        evidence_doc_recall=evidence_doc_recall,
        citation_precision=citation_precision,
        trace_completeness=trace_score,
        sufficiency_decision_score=suff_score,
        groundedness_proxy=ground_score,
        unsupported_penalty=unsupported_penalty,
        unsafe_omission_penalty=unsafe_penalty,
        msas=msas,
        trace_rate_limit_stats=(trace_summary or {}).get("rate_limit_stats", {}) if trace_summary else {},
        errors=errors or [],
        raw_response=response,
    )


def aggregate_run(run: RunAudit) -> RunAudit:
    if not run.cases:
        run.aggregate = {}
        return run
    numeric_fields = [
        "ndcg_at_10",
        "precision_at_10",
        "recall_at_10",
        "reciprocal_rank",
        "facet_recall",
        "critical_facet_recall",
        "evidence_doc_recall",
        "citation_precision",
        "trace_completeness",
        "sufficiency_decision_score",
        "groundedness_proxy",
        "unsupported_penalty",
        "unsafe_omission_penalty",
        "msas",
    ]
    agg: dict[str, float] = {}
    for field in numeric_fields:
        vals = [float(getattr(c, field)) for c in run.cases]
        agg[f"mean_{field}"] = sum(vals) / len(vals)
    passed = Counter(c.policy_passed for c in run.cases)
    agg["policy_pass_rate"] = passed.get(True, 0) / len(run.cases)
    agg["degraded_rate"] = sum(1 for c in run.cases if c.degraded_mode) / len(run.cases)
    agg["error_rate"] = sum(1 for c in run.cases if c.errors) / len(run.cases)
    buckets = Counter(c.failure_bucket for c in run.cases if c.failure_bucket)
    run.diagnostics["failure_buckets"] = dict(buckets)
    run.diagnostics["metric_semantics"] = {
        "trec_ranked_evidence": ["mean_ndcg_at_10", "mean_precision_at_10", "mean_recall_at_10", "mean_reciprocal_rank"],
        "system_audit_diagnostic": "mean_msas",
        "facet_warning": (
            "Automatically prepared TREC facets are seeded from document-level qrels; "
            "facet/sufficiency metrics remain provisional until facet-level adjudication."
        ),
        "groundedness_warning": (
            "groundedness_proxy checks citation/ledger alignment only; use clinician/rubric claim adjudication for publication claims."
        ),
    }
    run.aggregate = agg
    run.num_cases = len(run.cases)
    return run
