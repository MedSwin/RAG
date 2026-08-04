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
        }
        routed = None
        if "guideline_concordance" in missing or "evidence_quality" in missing:
            hints["focus_source"] = SourceType.CPG.value
            routed = "guideline" if "guideline_concordance" in missing else "quality"
        elif "patient_applicability" in missing:
            hints["focus_source"] = SourceType.EMR.value
            routed = "emr"
        if "safety_contraindications" in missing:
            hints["safety_search"] = True
            hints["relax_filters"] = True
            hints["focus_source"] = SourceType.SAFETY.value
            routed = "safety"
        if contradictions:
            hints["contradiction_review"] = True
            hints["relax_filters"] = True
            routed = "critic"
        hints["routed_agent"] = routed
        return hints

    def decide(
        self,
        ledger,
        facets: List[ClinicalFacet],
        passages: List[CandidatePassage],
        iteration: int,
        clinical_scope: ClinicalScope,
        contradictions: Optional[List[ContradictionPair]] = None,
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
        unresolved_critical = any(not item.resolved and item.severity == "high" for item in contradictions)
        marginal = estimate_marginal_utility_per_token(passages, coverage)
        can_retrieve_more = iteration < self.max_loops - 1
        contradiction_overflow = len(contradictions) > settings.SUFF_MAX_CONTRADICTIONS

        if not missing and not unresolved_critical and not contradiction_overflow:
            action = PolicyAction.ACCEPT
            reason = "All required clinical facets passed LCB, entropy, and contradiction gates."
            passed = True
        elif can_retrieve_more and (
            missing
            or unresolved_critical
            or marginal > settings.SUFF_MIN_MARGINAL_UTILITY
        ):
            action = PolicyAction.RETRIEVE_MORE
            reason = "Required facets, contradiction adjudication, or marginal utility justify targeted retrieval."
            passed = False
        else:
            action = PolicyAction.INSUFFICIENT_EVIDENCE
            reason = "Evidence remains insufficient for clinician CDS after policy-bounded retrieval."
            passed = False

        hints = self._retrieval_hints(missing, contradictions)
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
    ) -> SufficiencyCheck:
        constraints = constraints or {}
        scope = self._clinical_scope(constraints)
        if query_spec and query_spec.clinical_scope:
            scope = query_spec.clinical_scope
        decision = self.decide(ledger, facets, passages, iteration, scope)
        cpg = [p for p in passages if p.source_type == SourceType.CPG]
        emr = [p for p in passages if p.source_type == SourceType.EMR]
        confidences = [
            p.calibrated_score or p.rerank_score or p.fusion_score or p.dense_score
            for p in passages
            if (p.calibrated_score or p.rerank_score or p.fusion_score or p.dense_score) is not None
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
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
