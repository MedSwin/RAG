"""Compatibility shim around EvidenceGate for legacy tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.medswin.gate import EvidenceGate
from app.medswin.ledger import build_retrieval_ledger, detect_contradictions
from app.medswin.normalize import QueryNormalizer
from app.schemas.evidence import CandidatePassage, QuerySpec
from app.schemas.facets import ClinicalFacet
from app.schemas.traces import SufficiencyCheck


class EvidenceSufficiencyPolicy:
    """Legacy-compatible wrapper over EvidenceGate + ledger builders."""

    def __init__(self):
        self.gate = EvidenceGate()
        self.last_policy_decision = None
        self.last_evidence_ledger = []
        self.last_facet_coverage = []
        self.last_contradictions = []

    def build_facets(self, query, query_spec=None, constraints=None, patient_id=None):
        return QueryNormalizer.build_facets(query, query_spec, constraints, patient_id)

    def score_passage_facets(self, passage, facets):
        from app.scoring.coverage import score_passage_facets

        return score_passage_facets(passage, facets)

    def check_sufficiency(
        self,
        passages: List[CandidatePassage],
        iteration: int = 0,
        query_spec: Optional[QuerySpec] = None,
        constraints: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        selected_passages: Optional[List[CandidatePassage]] = None,
    ) -> SufficiencyCheck:
        facets = self.build_facets("", query_spec, constraints, patient_id)
        review = selected_passages or passages
        ledger = build_retrieval_ledger(review, facets)
        check = self.gate.check(ledger, facets, review, iteration, constraints or {}, query_spec)
        self.last_policy_decision = check.policy_decision
        self.last_evidence_ledger = ledger
        self.last_facet_coverage = check.facet_coverage
        self.last_contradictions = detect_contradictions(ledger)
        return check

    def should_retrieve_more(self, check: SufficiencyCheck) -> bool:
        return self.gate.should_retrieve_more(check)

    def get_retrieval_hints(self, check: SufficiencyCheck) -> Dict[str, Any]:
        if check.policy_decision:
            return check.policy_decision.retrieval_hints
        return {"increase_k": True}
