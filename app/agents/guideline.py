"""Guideline specialist agent."""

from __future__ import annotations

from typing import List, Optional

from app.agents.base import call_claim_agent
from app.prompts import claims as claim_prompts
from app.schemas.agents import AgentClaimBatch
from app.schemas.evidence import CandidatePassage
from app.schemas.facets import ClinicalFacet
from app.services.adapters.llm import LLMClient


class GuidelineAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    async def explore(
        self,
        query: str,
        passages: List[CandidatePassage],
        facets: Optional[List[ClinicalFacet]] = None,
    ) -> AgentClaimBatch:
        cpg = [p for p in passages if p.source_type.value in {"CPG", "LIT"}]
        facet_names = ", ".join(f.name for f in (facets or []))
        user = (
            f"Query: {query}\nTarget facets: {facet_names}\n"
            "Prioritise source_type CPG and LIT for recommendations and guideline concordance."
        )
        return await call_claim_agent(
            self.client, "guideline", claim_prompts.GUIDELINE, user, cpg, facets=facets, shared_passages=cpg
        )
