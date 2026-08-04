"""EMR specialist agent."""

from __future__ import annotations

from typing import List, Optional

from app.agents.base import call_claim_agent, passage_context
from app.prompts import claims as claim_prompts
from app.schemas.agents import AgentClaimBatch
from app.schemas.evidence import CandidatePassage
from app.schemas.facets import ClinicalFacet
from app.services.adapters.llm import LLMClient


class EMRAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    async def explore(
        self,
        query: str,
        passages: List[CandidatePassage],
        facets: Optional[List[ClinicalFacet]] = None,
        patient_id: Optional[str] = None,
    ) -> AgentClaimBatch:
        emr = [p for p in passages if p.source_type.value == "EMR"] or passages
        facet_names = ", ".join(f.name for f in (facets or []))
        user = (
            f"Query: {query}\nPatient ID: {patient_id or 'unknown'}\n"
            f"Target facets: {facet_names}\n\nPassages:\n{passage_context(emr)}"
        )
        return await call_claim_agent(self.client, "emr", claim_prompts.EMR, user, emr)
