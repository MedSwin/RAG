"""Evidence sufficiency policy with deterministic gates."""

import logging
from typing import Dict, List, Any, Optional
from app.core.config import settings
from app.models.medswin import CandidatePassage, SourceType, SufficiencyCheck
from datetime import datetime

logger = logging.getLogger(__name__)


class EvidenceSufficiencyPolicy:
    """Deterministic evidence sufficiency policy."""
    
    def __init__(self):
        """Initialize sufficiency policy with config values."""
        self.t_cpg = settings.SUFF_T_CPG
        self.t_emr = settings.SUFF_T_EMR
        self.t_inclusion = settings.SUFF_T_INCLUSION
        self.t_mean_conf = settings.SUFF_T_MEAN_CONF
        self.max_loops = settings.MAX_RETRIEVE_LOOPS
    
    def check_sufficiency(
        self,
        passages: List[CandidatePassage],
        iteration: int = 0
    ) -> SufficiencyCheck:
        """Check if evidence is sufficient.
        
        Args:
            passages: List of candidate passages
            iteration: Current iteration number
            
        Returns:
            SufficiencyCheck with pass/fail status and metrics
        """
        # Count passages by source type
        cpg_passages = [p for p in passages if p.source_type == SourceType.CPG]
        emr_passages = [p for p in passages if p.source_type == SourceType.EMR]
        
        # Compute coverage ratios
        kappa_cpg = len(cpg_passages) / self.t_cpg if self.t_cpg > 0 else 0.0
        kappa_emr = len(emr_passages) / self.t_emr if self.t_emr > 0 else 0.0
        
        # Compute mean confidence (from rerank scores or fusion scores)
        confidences = []
        for p in passages:
            if p.rerank_score is not None:
                confidences.append(p.rerank_score)
            elif p.fusion_score is not None:
                confidences.append(p.fusion_score)
            elif p.dense_score is not None:
                confidences.append(p.dense_score)
        
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Check sufficiency gates
        passed = (
            kappa_cpg >= 1.0 and
            kappa_emr >= 1.0 and
            mean_confidence >= self.t_mean_conf
        )
        
        # Determine action
        action_taken = None
        if not passed and iteration < self.max_loops:
            if kappa_cpg < 1.0:
                action_taken = "retrieve_more_cpg"
            elif kappa_emr < 1.0:
                action_taken = "retrieve_more_emr"
            elif mean_confidence < self.t_mean_conf:
                action_taken = "retrieve_more_high_confidence"
            else:
                action_taken = "retrieve_more"
        elif not passed:
            action_taken = "insufficient_evidence"
        
        return SufficiencyCheck(
            iteration=iteration,
            kappa_cpg=kappa_cpg,
            kappa_emr=kappa_emr,
            mean_confidence=mean_confidence,
            passed=passed,
            action_taken=action_taken,
            timestamp=datetime.utcnow()
        )
    
    def should_retrieve_more(self, check: SufficiencyCheck) -> bool:
        """Determine if we should retrieve more evidence."""
        return not check.passed and check.iteration < self.max_loops
    
    def get_retrieval_hints(self, check: SufficiencyCheck) -> Dict[str, Any]:
        """Get hints for next retrieval iteration.
        
        Returns:
            Dict with hints for increasing K, relaxing filters, etc.
        """
        hints = {
            "increase_k": True,
            "relax_filters": False,
            "expand_synonyms": False
        }
        
        if check.kappa_cpg < 1.0:
            hints["focus_source"] = SourceType.CPG.value
        elif check.kappa_emr < 1.0:
            hints["focus_source"] = SourceType.EMR.value
        
        if check.mean_confidence < self.t_mean_conf:
            hints["expand_synonyms"] = True
            hints["relax_filters"] = True
        
        return hints

