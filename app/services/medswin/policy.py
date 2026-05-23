"""Facet-level evidence sufficiency policy with deterministic gates."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.core.config import settings
from app.models.medswin import (
    CandidatePassage,
    ClinicalFacet,
    ClinicalScope,
    ContradictionPair,
    EvidenceClaim,
    EvidenceLedgerEntry,
    EvidencePolarity,
    FacetCoverage,
    PolicyAction,
    PolicyDecision,
    QuerySpec,
    SourceType,
    SufficiencyCheck,
)
from app.services.medswin.governance import clamp, evidence_grade_from_metadata
from facets import benchmark_required_facets

logger = logging.getLogger(__name__)


class EvidenceSufficiencyPolicy:
    """Enterprise evidence policy for clinician-facing MedSwin responses."""

    def __init__(self):
        """Initialize sufficiency policy with config values."""
        self.t_cpg = settings.SUFF_T_CPG
        self.t_emr = settings.SUFF_T_EMR
        self.t_inclusion = settings.SUFF_T_INCLUSION
        self.t_mean_conf = settings.SUFF_T_MEAN_CONF
        self.max_loops = settings.MAX_RETRIEVE_LOOPS
        self.last_policy_decision: Optional[PolicyDecision] = None
        self.last_evidence_ledger: List[EvidenceLedgerEntry] = []
        self.last_facet_coverage: List[FacetCoverage] = []
        self.last_contradictions: List[ContradictionPair] = []

    def build_facets(
        self,
        query: str,
        query_spec: Optional[QuerySpec] = None,
        constraints: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
    ) -> List[ClinicalFacet]:
        """Build query-specific clinical facets from constraints and normalized query."""
        constraints = constraints or {}
        explicit_facets = constraints.get("required_facets") or []
        if explicit_facets:
            return [self._coerce_facet(item) for item in benchmark_required_facets(None, explicit_facets)]

        if query_spec and query_spec.facets:
            return query_spec.facets

        threshold = settings.SUFF_CRITICAL_FACET_THRESHOLD
        facets = [
            ClinicalFacet(
                name="guideline_concordance",
                required=True,
                threshold=threshold,
                weight=1.20,
                source_policy="CPG",
                keywords=["guideline", "recommendation", "indication", "management", "treatment"],
            ),
            ClinicalFacet(
                name="safety_contraindications",
                required=True,
                threshold=threshold,
                weight=1.35,
                source_policy="ANY",
                keywords=["contraindication", "avoid", "adverse", "risk", "allergy", "interaction"],
            ),
        ]

        patient_required = bool(patient_id) or "patient" in query.lower() or "elderly" in query.lower()
        facets.append(
            ClinicalFacet(
                name="patient_applicability",
                required=patient_required,
                threshold=settings.SUFF_FACET_THRESHOLD,
                weight=1.05,
                source_policy="EMR" if patient_required else "ANY",
                keywords=["patient", "history", "medication", "lab", "allergy", "comorbidity", "age"],
            )
        )
        facets.append(
            ClinicalFacet(
                name="evidence_quality",
                required=True,
                threshold=settings.SUFF_FACET_THRESHOLD,
                weight=0.95,
                source_policy="ANY",
                keywords=["grade", "evidence", "trial", "review", "recommendation", "version"],
            )
        )
        return facets

    def check_sufficiency(
        self,
        passages: List[CandidatePassage],
        iteration: int = 0,
        query_spec: Optional[QuerySpec] = None,
        constraints: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        selected_passages: Optional[List[CandidatePassage]] = None,
    ) -> SufficiencyCheck:
        """Check whether evidence is sufficient for clinician CDS generation."""
        constraints = constraints or {}
        facets = self.build_facets("", query_spec, constraints, patient_id)
        review_passages = selected_passages or passages
        ledger = self.build_evidence_ledger(review_passages, facets)
        coverage = self.compute_facet_coverage(facets, ledger)
        contradictions = self.detect_contradictions(ledger)
        decision = self.make_policy_decision(
            passages=review_passages,
            facets=facets,
            coverage=coverage,
            contradictions=contradictions,
            iteration=iteration,
            clinical_scope=self._clinical_scope(constraints),
        )

        self.last_policy_decision = decision
        self.last_evidence_ledger = ledger
        self.last_facet_coverage = coverage
        self.last_contradictions = contradictions

        cpg_passages = [p for p in review_passages if p.source_type == SourceType.CPG]
        emr_passages = [p for p in review_passages if p.source_type == SourceType.EMR]
        kappa_cpg = len(cpg_passages) / self.t_cpg if self.t_cpg > 0 else 0.0
        kappa_emr = len(emr_passages) / self.t_emr if self.t_emr > 0 else 0.0
        confidences = [
            p.calibrated_score or p.rerank_score or p.fusion_score or p.dense_score
            for p in review_passages
            if (p.calibrated_score or p.rerank_score or p.fusion_score or p.dense_score) is not None
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return SufficiencyCheck(
            iteration=iteration,
            kappa_cpg=kappa_cpg,
            kappa_emr=kappa_emr,
            mean_confidence=mean_confidence,
            passed=decision.passed,
            action_taken=decision.action.value,
            facet_coverage=coverage,
            contradiction_count=len(contradictions),
            missing_facets=decision.missing_facets,
            marginal_utility_per_token=decision.marginal_utility_per_token,
            policy_decision=decision,
            timestamp=datetime.utcnow(),
        )

    def build_evidence_ledger(
        self,
        passages: List[CandidatePassage],
        facets: List[ClinicalFacet],
        agent_id: str = "retrieval",
    ) -> List[EvidenceLedgerEntry]:
        """Build a claim-level ledger from candidate passages.

        Motivation vs Logic: MedSwin.tex treats evidence as typed claims with provenance,
        not an undifferentiated context blob. The runtime therefore maps each passage to
        facet claims, calibrated relevance, EBM grade, and contradiction/safety signals
        before any answer can be accepted.
        """
        ledger: List[EvidenceLedgerEntry] = []
        for passage in passages:
            grade = evidence_grade_from_metadata(passage)
            calibrated = clamp(passage.calibrated_score or passage.rerank_score or passage.fusion_score or 0.0)
            facet_scores = self.score_passage_facets(passage, facets)
            claims = []
            matched_facets = []
            for facet in facets:
                score = facet_scores.get(facet.name, 0.0)
                if score <= 0.0:
                    continue
                polarity = self._infer_polarity(passage, facet)
                matched_facets.append(facet.name)
                claims.append(
                    EvidenceClaim(
                        facet=facet.name,
                        claim=self._claim_excerpt(passage.text),
                        polarity=polarity,
                        chunk_id=passage.chunk_id,
                        confidence=clamp(score * max(calibrated, 0.35) * grade.score),
                        evidence_grade=grade,
                        provenance=self._provenance(passage),
                    )
                )

            ledger.append(
                EvidenceLedgerEntry(
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                    agent_id=agent_id,
                    facets=matched_facets,
                    claims=claims,
                    calibrated_relevance=calibrated,
                    fusion_score=clamp(passage.fusion_score or calibrated),
                    evidence_grade=grade,
                    safety_relevance=clamp(passage.safety_score or self._safety_score(passage)),
                    contradiction_risk=clamp(passage.contradiction_score or self._contradiction_risk(passage)),
                    provenance=self._provenance(passage),
                )
            )
        return ledger

    def score_passage_facets(
        self,
        passage: CandidatePassage,
        facets: Iterable[ClinicalFacet],
    ) -> Dict[str, float]:
        """Score passage-to-facet alignment with metadata first, then heuristics."""
        text = passage.text.lower()
        explicit_scores = passage.facet_scores or passage.metadata.get("facet_scores", {})
        scores: Dict[str, float] = {}
        for facet in facets:
            if facet.name in explicit_scores:
                scores[facet.name] = clamp(explicit_scores[facet.name])
                continue

            source_bonus = self._source_policy_score(passage.source_type, facet.source_policy)
            keyword_hits = sum(1 for keyword in facet.keywords if keyword.lower() in text)
            keyword_score = min(0.65, keyword_hits * 0.18)
            section_score = passage.section_score if passage.section_score is not None else 0.5
            safety_bonus = 0.25 if facet.name == "safety_contraindications" and self._safety_score(passage) > 0.0 else 0.0
            if source_bonus == 0.0 and keyword_hits == 0 and safety_bonus == 0.0:
                scores[facet.name] = 0.0
            else:
                scores[facet.name] = clamp(0.20 + source_bonus + keyword_score + 0.15 * section_score + safety_bonus)
        passage.facet_scores = scores
        return scores

    def compute_facet_coverage(
        self,
        facets: List[ClinicalFacet],
        ledger: List[EvidenceLedgerEntry],
    ) -> List[FacetCoverage]:
        """Compute noisy-OR coverage with a conservative lower confidence bound."""
        coverage: List[FacetCoverage] = []
        for facet in facets:
            no_support_probability = 1.0
            supporting: List[str] = []
            contradicting: List[str] = []
            for entry in ledger:
                for claim in entry.claims:
                    if claim.facet != facet.name:
                        continue
                    contribution = clamp(claim.confidence)
                    if claim.polarity == EvidencePolarity.CONTRADICTS:
                        contradicting.append(entry.chunk_id)
                        contribution *= 0.35
                    else:
                        supporting.append(entry.chunk_id)
                    no_support_probability *= 1.0 - contribution
            probability = clamp(1.0 - no_support_probability)
            lcb = clamp(probability - settings.SUFF_LCB_MARGIN)
            entropy = self._binary_entropy(probability)
            if lcb >= facet.threshold and entropy <= settings.SUFF_MAX_ENTROPY:
                status = "satisfied"
            elif supporting:
                status = "uncertain"
            else:
                status = "missing"
            coverage.append(
                FacetCoverage(
                    facet=facet.name,
                    required=facet.required,
                    threshold=facet.threshold,
                    coverage_probability=probability,
                    lower_confidence_bound=lcb,
                    entropy=entropy,
                    status=status,
                    supporting_chunk_ids=sorted(set(supporting)),
                    contradicting_chunk_ids=sorted(set(contradicting)),
                )
            )
        return coverage

    def detect_contradictions(self, ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
        """Detect unresolved high-risk contradiction pairs by facet."""
        by_facet: Dict[str, Dict[str, List[EvidenceLedgerEntry]]] = {}
        for entry in ledger:
            for claim in entry.claims:
                bucket = by_facet.setdefault(claim.facet, {"support": [], "contradict": []})
                if claim.polarity == EvidencePolarity.CONTRADICTS:
                    bucket["contradict"].append(entry)
                elif claim.polarity in {EvidencePolarity.SUPPORTS, EvidencePolarity.QUALIFIES, EvidencePolarity.SAFETY}:
                    bucket["support"].append(entry)

        contradictions: List[ContradictionPair] = []
        for facet, bucket in by_facet.items():
            for support in bucket["support"]:
                for conflict in bucket["contradict"]:
                    if support.chunk_id == conflict.chunk_id:
                        continue
                    severity = "high" if support.evidence_grade.score >= 0.80 or conflict.evidence_grade.score >= 0.80 else "medium"
                    contradictions.append(
                        ContradictionPair(
                            facet=facet,
                            chunk_id_a=support.chunk_id,
                            chunk_id_b=conflict.chunk_id,
                            severity=severity,
                            reason="High-grade evidence contains incompatible support and caution for the same facet.",
                        )
                    )
        return contradictions

    def make_policy_decision(
        self,
        passages: List[CandidatePassage],
        facets: List[ClinicalFacet],
        coverage: List[FacetCoverage],
        contradictions: List[ContradictionPair],
        iteration: int,
        clinical_scope: ClinicalScope,
    ) -> PolicyDecision:
        """Convert facet coverage into an accept/retrieve/fail decision."""
        missing = [
            item.facet
            for item in coverage
            if item.required and (
                item.lower_confidence_bound < item.threshold or item.entropy > settings.SUFF_MAX_ENTROPY
            )
        ]
        unresolved_critical = any(not item.resolved and item.severity == "high" for item in contradictions)
        marginal = self.estimate_marginal_utility_per_token(passages, coverage)
        can_retrieve_more = iteration < self.max_loops - 1

        if not missing and not unresolved_critical and len(contradictions) <= settings.SUFF_MAX_CONTRADICTIONS:
            action = PolicyAction.ACCEPT
            reason = "All required clinical facets passed calibrated coverage, entropy, and contradiction gates."
            passed = True
        elif can_retrieve_more and (missing or unresolved_critical or marginal > settings.SUFF_MIN_MARGINAL_UTILITY):
            action = PolicyAction.RETRIEVE_MORE
            reason = "Required facets, contradiction adjudication, or marginal utility justify targeted retrieval."
            passed = False
        else:
            action = PolicyAction.INSUFFICIENT_EVIDENCE
            reason = "Evidence remains insufficient for clinician CDS after policy-bounded retrieval."
            passed = False

        hints = self._retrieval_hints(missing, contradictions)
        return PolicyDecision(
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
        )

    def estimate_marginal_utility_per_token(
        self,
        passages: List[CandidatePassage],
        coverage: List[FacetCoverage],
    ) -> float:
        """Estimate residual retrieval value per evidence token."""
        if not passages:
            return 1.0
        deficit = sum(max(0.0, item.threshold - item.lower_confidence_bound) for item in coverage if item.required)
        total_tokens = sum(p.token_count or max(1, len(p.text.split())) for p in passages)
        return deficit / max(total_tokens, 1)

    def should_retrieve_more(self, check: SufficiencyCheck) -> bool:
        """Determine if another targeted retrieval iteration is allowed."""
        return (
            not check.passed
            and check.iteration < self.max_loops - 1
            and check.action_taken == PolicyAction.RETRIEVE_MORE.value
        )

    def get_retrieval_hints(self, check: SufficiencyCheck) -> Dict[str, Any]:
        """Get next-iteration retrieval hints from the current policy decision."""
        if check.policy_decision:
            return check.policy_decision.retrieval_hints
        hints = {"increase_k": True, "relax_filters": False, "expand_synonyms": False}
        if check.missing_facets:
            hints["missing_facets"] = check.missing_facets
        return hints

    def _coerce_facet(self, item: Any) -> ClinicalFacet:
        if isinstance(item, ClinicalFacet):
            return item
        if isinstance(item, dict):
            return ClinicalFacet(
                threshold=item.get("threshold", settings.SUFF_CRITICAL_FACET_THRESHOLD if item.get("required", True) else settings.SUFF_FACET_THRESHOLD),
                **{key: value for key, value in item.items() if key != "threshold"},
            )
        return ClinicalFacet(name=str(item), threshold=settings.SUFF_CRITICAL_FACET_THRESHOLD)

    def _clinical_scope(self, constraints: Dict[str, Any]) -> ClinicalScope:
        try:
            return ClinicalScope(constraints.get("clinical_scope") or settings.DEFAULT_CLINICAL_SCOPE)
        except ValueError:
            return ClinicalScope.CLINICIAN_CDS

    def _source_policy_score(self, source_type: SourceType, source_policy: Optional[str]) -> float:
        if not source_policy:
            return 0.0
        if source_policy == "ANY":
            return 0.20
        if source_policy == source_type.value:
            return 0.45
        if source_policy == "CPG" and source_type == SourceType.LIT:
            return 0.15
        return 0.0

    def _infer_polarity(self, passage: CandidatePassage, facet: ClinicalFacet) -> EvidencePolarity:
        explicit = passage.metadata.get("polarity")
        if explicit:
            try:
                return EvidencePolarity(explicit)
            except ValueError:
                pass
        text = passage.text.lower()
        if any(term in text for term in ["contraindicat", "not recommended", "avoid", "do not", "should not"]):
            return EvidencePolarity.CONTRADICTS if facet.name != "safety_contraindications" else EvidencePolarity.SAFETY
        if any(term in text for term in ["unless", "except", "caution", "monitor"]):
            return EvidencePolarity.QUALIFIES
        return EvidencePolarity.SUPPORTS

    def _safety_score(self, passage: CandidatePassage) -> float:
        text = passage.text.lower()
        safety_terms = ["contraindicat", "allergy", "adverse", "interaction", "avoid", "risk", "toxicity", "dose"]
        return clamp(sum(1 for term in safety_terms if term in text) * 0.18)

    def _contradiction_risk(self, passage: CandidatePassage) -> float:
        text = passage.text.lower()
        risk_terms = ["conflict", "not recommended", "avoid", "insufficient", "uncertain", "contraindicat"]
        return clamp(sum(1 for term in risk_terms if term in text) * 0.16)

    def _claim_excerpt(self, text: str) -> str:
        compact = " ".join(text.split())
        return compact[:280]

    def _provenance(self, passage: CandidatePassage) -> Dict[str, Any]:
        return {
            "doc_id": passage.doc_id,
            "section": passage.section,
            "offset_start": passage.offset_start,
            "offset_end": passage.offset_end,
            "source_type": passage.source_type.value,
            "guideline_version": passage.metadata.get("guideline_version") or passage.metadata.get("version"),
            "effective_date": passage.metadata.get("effective_date"),
            "timestamp": passage.metadata.get("timestamp"),
        }

    def _binary_entropy(self, probability: float) -> float:
        p = clamp(probability, 1e-9, 1.0 - 1e-9)
        return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    def _retrieval_hints(
        self,
        missing_facets: List[str],
        contradictions: List[ContradictionPair],
    ) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "increase_k": True,
            "relax_filters": False,
            "expand_synonyms": bool(missing_facets),
            "missing_facets": missing_facets,
        }
        if "guideline_concordance" in missing_facets or "evidence_quality" in missing_facets:
            hints["focus_source"] = SourceType.CPG.value
        elif "patient_applicability" in missing_facets:
            hints["focus_source"] = SourceType.EMR.value
        if "safety_contraindications" in missing_facets:
            hints["safety_search"] = True
            hints["relax_filters"] = True
        if contradictions:
            hints["contradiction_review"] = True
            hints["relax_filters"] = True
        return hints
