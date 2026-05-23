from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import BenchmarkCase, CaseAudit, GoldFacet, RunAudit


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


def facet_recall(gold_facets: list[GoldFacet], evidence_doc_ids: set[str], *, critical_only: bool = False) -> float:
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
        # If a facet has no doc-level labels, do not penalize it in this automatic pass.
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
        # Root Cause vs Logic: the runtime emits plural count fields (for example
        # messages_count) while older benchmark code expected singular aliases. The
        # logic now accepts both so trace completeness reflects the actual audit payload.
        checks["trace_counts"] = (
            _trace_count(trace_summary, "messages_count", "message_count")
            or _trace_count(trace_summary, "tool_calls_count", "tool_count")
            or _trace_count(trace_summary, "sufficiency_checks_count", "sufficiency_check_count")
            or _trace_count(trace_summary, "evidence_passages_count", "evidence_count")
        )
    return sum(checks.values()) / len(checks)


def groundedness_proxy(response: dict[str, Any], cited_doc_ids: set[str]) -> tuple[float, float]:
    """Estimate groundedness from citations/ledger without an LLM judge.

    For publication, replace or supplement this with blinded clinician or rubric-based
    claim adjudication. The proxy is intentionally conservative: answers with no
    citations or no evidence ledger lose points.
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
            claim = item.get("claim") or item.get("text") or item.get("statement")
            if claim:
                total += 1
                doc_id = _norm_id(item.get("doc_id") or item.get("document_id") or item.get("source_id"))
                polarity = str(item.get("polarity", "support")).lower()
                if doc_id in cited_doc_ids and polarity in {"support", "supports", "qualifies"}:
                    supported += 1
        if total:
            score = supported / total
            return score, 1.0 - score
    # Fallback: citation presence per answer length.
    if citations:
        return 0.65, 0.35
    return 0.25, 0.75


def audit_case(case: BenchmarkCase, response: dict[str, Any], trace_summary: dict[str, Any] | None = None, errors: list[str] | None = None) -> CaseAudit:
    selected_doc_ids, cited_doc_ids, selected_chunk_ids = extract_doc_ids(response)
    evidence_doc_ids = set(selected_doc_ids) | set(cited_doc_ids)
    gold_doc_ids = set(case.gold_doc_ids)

    evidence_doc_recall = len(gold_doc_ids & evidence_doc_ids) / len(gold_doc_ids) if gold_doc_ids else 1.0
    citation_precision = len(set(cited_doc_ids) & gold_doc_ids) / len(set(cited_doc_ids)) if cited_doc_ids and gold_doc_ids else (1.0 if cited_doc_ids else 0.0)

    f_recall = facet_recall(case.gold_facets, evidence_doc_ids, critical_only=False)
    cf_recall = facet_recall(case.gold_facets, evidence_doc_ids, critical_only=True)

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

    clinical_quality_proxy = 0.5 * ground_score + 0.5 * min(1.0, len(str(response.get("answer") or "")) / 600.0)
    msas = (
        0.25 * cf_recall
        + 0.15 * f_recall
        + 0.15 * ground_score
        + 0.15 * suff_score
        + 0.10 * trace_score
        + 0.10 * evidence_doc_recall
        + 0.10 * clinical_quality_proxy
        - 0.20 * unsafe_penalty
        - 0.10 * unsupported_penalty
    )
    msas = max(0.0, min(1.0, msas))

    return CaseAudit(
        case_id=case.case_id,
        dataset=case.dataset,
        trace_id=response.get("trace_id") or (response.get("trace") or {}).get("trace_id"),
        session_id=response.get("session_id"),
        policy_passed=policy_passed,
        degraded_mode=bool(response.get("degraded_mode")) if response.get("degraded_mode") is not None else None,
        answer_chars=len(str(response.get("answer") or "")),
        selected_doc_ids=selected_doc_ids,
        cited_doc_ids=cited_doc_ids,
        selected_chunk_ids=selected_chunk_ids,
        gold_doc_ids=sorted(gold_doc_ids),
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
    run.aggregate = agg
    run.num_cases = len(run.cases)
    return run
