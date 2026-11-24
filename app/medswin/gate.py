"""Evidence-sufficiency gate (Answer / Retrieve-More / Insufficient)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.schemas.enums import ClinicalScope, PolicyAction, SourceType
from app.schemas.evidence import (
    CandidatePassage,
    ContradictionPair,
    PolicyDecision,
    QuerySpec,
    SufficiencyDecision,
)
from app.schemas.facets import ClinicalFacet, FacetCoverage
from app.schemas.traces import SufficiencyCheck
from app.scoring.coverage import compute_facet_coverage
from app.scoring.utility import estimate_marginal_utility_per_token
from app.medswin.ledger import detect_contradictions


def _passage_confidence(passage: CandidatePassage) -> Optional[float]:
    """Use the calibrated score only. A fusion or dense fallback must not clear 0.55."""
    if passage.calibrated_score is None:
        return None
    return float(passage.calibrated_score)


def _confidence_vote(passages: List[CandidatePassage]) -> tuple[bool, bool, float]:
    """Return inclusion pass, mean-confidence pass, and the mean score."""
    scores = [_passage_confidence(passage) for passage in passages]
    known = [score for score in scores if score is not None]
    mean_confidence = sum(known) / len(known) if known else 0.0
    inclusion_ok = bool(passages) and all(
        score is not None and score >= settings.SUFF_T_INCLUSION for score in scores
    )
    mean_ok = bool(passages) and mean_confidence >= settings.SUFF_T_MEAN_CONF
    return inclusion_ok, mean_ok, mean_confidence


def _confidence_gap(
    passages: List[CandidatePassage],
    inclusion_ok: bool,
    mean_ok: bool,
    mean_confidence: float,
) -> Optional[str]:
    """Name the confidence failure. An empty set stays on the provenance path."""
    if not passages:
        return None
    if not inclusion_ok:
        return f"A passage is below the inclusion threshold {settings.SUFF_T_INCLUSION:.2f}."
    if not mean_ok:
        return (
            f"Mean passage confidence {mean_confidence:.2f} is below "
            f"{settings.SUFF_T_MEAN_CONF:.2f}."
        )
    return None


class EvidenceGate:
    """Paper sufficiency contract with bootstrap LCB + entropy + contradiction + ε_U."""

    def __init__(self):
        self.max_loops = settings.MAX_RETRIEVE_LOOPS
        self.last_decision: Optional[PolicyDecision] = None
        self.last_coverage: List[FacetCoverage] = []
        self.last_contradictions: List[ContradictionPair] = []

    def _clinical_scope(self, constraints: Dict[str, Any]) -> ClinicalScope:
        try:
            return ClinicalScope(constraints.get("clinical_scope") or settings.DEFAULT_CLINICAL_SCOPE)
        except ValueError:
            return ClinicalScope.CLINICIAN_CDS

    def _retrieval_hints(
        self,
        missing: List[str],
        contradictions: List[ContradictionPair],
    ) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "increase_k": True,
            "relax_filters": False,
            "expand_synonyms": bool(missing),
            "missing_facets": missing,
            "exclusive_source": False,
        }
        routed: List[str] = []
        sources: List[str] = []
        if "guideline_concordance" in missing:
            routed.append("guideline")
            sources.append(SourceType.CPG.value)
        if "evidence_quality" in missing:
            routed.append("quality")
            sources.append(SourceType.CPG.value)
        if "patient_applicability" in missing:
            routed.append("emr")
            sources.append(SourceType.EMR.value)
        if "safety_contraindications" in missing:
            hints["safety_search"] = True
            hints["relax_filters"] = True
            routed.append("safety")
            sources.append(SourceType.SAFETY.value)
        if any(not item.resolved for item in contradictions):
            hints["contradiction_review"] = True
            hints["relax_filters"] = True
            routed.append("critic")
        unique_sources = list(dict.fromkeys(sources))
        if len(unique_sources) == 1:
            hints["focus_source"] = unique_sources[0]
            hints["exclusive_source"] = True
        elif unique_sources:
            hints["focus_source"] = unique_sources[0]
            hints["exclusive_source"] = False
        hints["routed_agent"] = routed[0] if routed else None
        hints["routed_agents"] = routed
        return hints

    def decide(
        self,
        ledger,
        facets: List[ClinicalFacet],
        passages: List[CandidatePassage],
        iteration: int,
        clinical_scope: ClinicalScope,
        contradictions: Optional[List[ContradictionPair]] = None,
        stop_retrieval: bool = False,
    ) -> PolicyDecision:
        coverage = compute_facet_coverage(facets, ledger)
        contradictions = contradictions if contradictions is not None else detect_contradictions(ledger)
        missing = [
            item.facet
            for item in coverage
            if item.required
            and (
                item.lower_confidence_bound < item.threshold
                or item.entropy > settings.SUFF_MAX_ENTROPY
            )
        ]
        unresolved = [item for item in contradictions if not item.resolved]
        unresolved_critical = any(item.severity == "high" for item in unresolved)
        marginal = estimate_marginal_utility_per_token(passages, coverage)
        can_retrieve_more = iteration < self.max_loops - 1 and not stop_retrieval
        contradiction_overflow = len(unresolved) > settings.SUFF_MAX_CONTRADICTIONS
        provenance_ok = bool(passages) and all(
            bool(p.chunk_id)
            and bool(p.doc_id)
            and bool(p.source_type)
            and not (p.metadata or {}).get("weak_provenance")
            for p in passages
        )
        inclusion_ok, mean_ok, mean_confidence = _confidence_vote(passages)
        confidence_gap = _confidence_gap(passages, inclusion_ok, mean_ok, mean_confidence)
        bundle_ok = (
            not missing
            and not unresolved_critical
            and not contradiction_overflow
            and provenance_ok
        )

        if bundle_ok and confidence_gap is None:
            action = PolicyAction.ACCEPT
            reason = (
                "All required clinical facets passed LCB, entropy, contradiction, "
                "inclusion, and mean-confidence gates."
            )
            passed = True
        elif can_retrieve_more and (
            missing
            or unresolved_critical
            or contradiction_overflow
            or marginal > settings.SUFF_MIN_MARGINAL_UTILITY
            or confidence_gap is not None
        ):
            action = PolicyAction.RETRIEVE_MORE
            reason = (
                confidence_gap
                if bundle_ok and confidence_gap
                else "Required facets, contradiction adjudication, or marginal utility justify targeted retrieval."
            )
            passed = False
        else:
            action = PolicyAction.INSUFFICIENT_EVIDENCE
            reason = (
                confidence_gap
                if bundle_ok and confidence_gap
                else "Evidence remains insufficient for clinician CDS after policy-bounded retrieval."
            )
            passed = False

        hints = self._retrieval_hints(missing, contradictions)
        hints["mean_confidence"] = round(mean_confidence, 4)
        hints["inclusion_threshold"] = settings.SUFF_T_INCLUSION
        hints["mean_confidence_threshold"] = settings.SUFF_T_MEAN_CONF
        hints["inclusion_passed"] = inclusion_ok
        hints["mean_confidence_passed"] = mean_ok
        decision = PolicyDecision(
            passed=passed,
            action=action,
            reason=reason,
            iteration=iteration,
            clinical_scope=clinical_scope,
            facet_coverage=coverage,
            contradictions=contradictions,
            marginal_utility_per_token=marginal,
            unresolved_critical_conflicts=unresolved_critical,
            missing_facets=missing,
            retrieval_hints=hints,
            routed_agent=hints.get("routed_agent"),
        )
        self.last_decision = decision
        self.last_coverage = coverage
        self.last_contradictions = contradictions
        return decision

    def check(
        self,
        ledger,
        facets: List[ClinicalFacet],
        passages: List[CandidatePassage],
        iteration: int = 0,
        constraints: Optional[Dict[str, Any]] = None,
        query_spec: Optional[QuerySpec] = None,
        contradictions: Optional[List[ContradictionPair]] = None,
        stop_retrieval: bool = False,
    ) -> SufficiencyCheck:
        constraints = constraints or {}
        scope = self._clinical_scope(constraints)
        if query_spec and query_spec.clinical_scope:
            scope = query_spec.clinical_scope
        decision = self.decide(
            ledger,
            facets,
            passages,
            iteration,
            scope,
            contradictions=contradictions,
            stop_retrieval=stop_retrieval,
        )
        cpg = [p for p in passages if p.source_type == SourceType.CPG]
        emr = [p for p in passages if p.source_type == SourceType.EMR]
        recorded_mean = (decision.retrieval_hints or {}).get("mean_confidence")
        mean_confidence = float(recorded_mean) if recorded_mean is not None else 0.0
        return SufficiencyCheck(
            iteration=iteration,
            kappa_cpg=len(cpg) / max(settings.SUFF_T_CPG, 1),
            kappa_emr=len(emr) / max(settings.SUFF_T_EMR, 1),
            mean_confidence=mean_confidence,
            passed=decision.passed,
            action_taken=decision.action.value,
            facet_coverage=decision.facet_coverage,
            contradiction_count=len(decision.contradictions),
            missing_facets=decision.missing_facets,
            marginal_utility_per_token=decision.marginal_utility_per_token,
            policy_decision=decision,
            timestamp=datetime.utcnow(),
        )

    def to_sufficiency_decision(self, decision: PolicyDecision) -> SufficiencyDecision:
        return SufficiencyDecision(
            passed=decision.passed,
            action=decision.action,
            reason=decision.reason,
            iteration=decision.iteration,
            missing_facets=decision.missing_facets,
            unresolved_critical_conflicts=decision.unresolved_critical_conflicts,
            marginal_utility_per_token=decision.marginal_utility_per_token,
            routed_agent=decision.routed_agent,
        )

    def should_retrieve_more(self, check: SufficiencyCheck) -> bool:
        return (
            not check.passed
            and check.iteration < self.max_loops - 1
            and check.action_taken == PolicyAction.RETRIEVE_MORE.value
        )
