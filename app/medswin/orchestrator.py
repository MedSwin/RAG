"""MedSwin orchestrator: MAC loop with evidence-sufficiency gate."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.critic import ContradictionAgent
from app.agents.emr import EMRAgent
from app.agents.guideline import GuidelineAgent
from app.agents.quality import QualityAgent
from app.agents.safety import SafetyAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.weights import ReliabilityWeights
from app.core.config import settings
from app.medswin.abstain import insufficient_answer
from app.medswin.gate import EvidenceGate
from app.medswin.ledger import build_retrieval_ledger, detect_contradictions, merge_agent_claims
from app.medswin.normalize import QueryNormalizer
from app.medswin.packing import pack_bundle
from app.repositories.chunks import ChunkRepository
from app.repositories.sessions import SessionRepository
from app.repositories.traces import TraceRepository
from app.retrieval.hybrid import HybridRetriever
from app.schemas.enums import ClinicalScope, PolicyAction, SourceType
from app.schemas.evidence import ContradictionLedger, EvidenceBundle, QuerySpec
from app.schemas.facets import FacetMatrix
from app.schemas.sessions import Session
from app.schemas.traces import AgentMessage, AuditTrace, ChatResponse, ToolCall
from app.scoring.calibrate import get_calibration_store
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient
from app.services.adapters.reranker import RerankerClient
from app.services.medswin.governance import build_citation, redact_phi_text

logger = logging.getLogger(__name__)


class MedSwinOrchestrator:
    """Centralized orchestrator-mediated multi-agent CDSS runtime."""

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
    ):
        cloud_model = settings.CLOUD_MODEL if settings.CLOUD_MODE else None
        self.supervisor = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
        self.emr_llm = LLMClient(settings.active_llm_url(settings.AGENT2_URL), model=cloud_model)
        self.guideline_llm = LLMClient(settings.active_llm_url(settings.AGENT3_URL), model=cloud_model)
        self.safety_llm = LLMClient(settings.active_llm_url(settings.AGENT1_URL), model=cloud_model)
        self.quality_llm = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
        self.critic_llm = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)

        self.embedding_client = embedding_client or EmbeddingClient(settings.active_embedding_url())
        self.reranker_client = reranker_client or RerankerClient(settings.active_reranker_url())
        self.retriever = HybridRetriever(reranker_client=self.reranker_client)
        self.gate = EvidenceGate()
        self.normalizer = QueryNormalizer(self.supervisor)
        self.weights = ReliabilityWeights()
        self.calibration = get_calibration_store()

        self.emr_agent = EMRAgent(self.emr_llm)
        self.guideline_agent = GuidelineAgent(self.guideline_llm)
        self.safety_agent = SafetyAgent(self.safety_llm)
        self.quality_agent = QualityAgent(self.quality_llm)
        self.critic_agent = ContradictionAgent(self.critic_llm)
        self.synthesis_agent = SynthesisAgent(self.supervisor)

        self.chunk_repo = ChunkRepository()
        self.session_repo = SessionRepository()
        self.trace_repo = TraceRepository()

    async def chat(
        self,
        query: str,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        trace_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
            await self.session_repo.create(Session(session_id=session_id, user_id=user_id, org_id=org_id), org_id)
        else:
            await self.session_repo.update_last_active(session_id, org_id)

        trace = AuditTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            query=query,
            patient_id=patient_id,
        )
        degraded: Dict[str, bool] = {}
        if not self.calibration.loaded:
            degraded["calibration"] = True

        try:
            query_spec = await self.normalizer.normalize(query, trace)
            facets = self.normalizer.build_facets(query, query_spec, constraints, patient_id)
            query_spec.facets = facets
            if constraints and constraints.get("clinical_scope"):
                try:
                    query_spec.clinical_scope = ClinicalScope(constraints["clinical_scope"])
                except ValueError:
                    query_spec.clinical_scope = ClinicalScope.CLINICIAN_CDS

            evidence_bundle, final_check = await self._retrieve_with_sufficiency(
                query=query,
                query_spec=query_spec,
                org_id=org_id,
                patient_id=patient_id,
                constraints=constraints or {},
                trace=trace,
            )

            decision = evidence_bundle.policy_decision
            facet_matrix = FacetMatrix(
                query=query,
                iteration=final_check.iteration if final_check else 0,
                rows=evidence_bundle.facet_coverage,
            )
            contradiction_ledger = ContradictionLedger(
                pairs=evidence_bundle.contradictions,
                unresolved_high=sum(
                    1 for p in evidence_bundle.contradictions if p.severity == "high" and not p.resolved
                ),
            )
            sufficiency_decision = self.gate.to_sufficiency_decision(decision) if decision else None
            trace.facet_matrix = facet_matrix
            trace.contradiction_ledger = contradiction_ledger
            trace.sufficiency_decision = sufficiency_decision
            trace.evidence_bundle = evidence_bundle
            trace.facet_coverage = evidence_bundle.facet_coverage
            trace.contradictions = evidence_bundle.contradictions
            trace.evidence_ledger = evidence_bundle.evidence_ledger
            trace.degraded_mode = degraded

            # P0: never synthesize on an empty accepted bundle
            if decision and decision.passed and not evidence_bundle.passages:
                decision.passed = False
                decision.action = PolicyAction.INSUFFICIENT_EVIDENCE
                decision.reason = "Gate passed without selectable passages; treating as insufficient evidence."
                evidence_bundle.policy_decision = decision

            if decision and not decision.passed:
                answer = insufficient_answer(query, decision, evidence_bundle.facet_coverage)
                citations = self._citations(evidence_bundle)
                trace.completed_at = datetime.utcnow()
                trace.final_answer = answer
                trace.citations = citations
                await self.trace_repo.create(trace, org_id)
                return ChatResponse(
                    answer=answer,
                    evidence_bundle=evidence_bundle,
                    safety_notes=decision.reason,
                    trace_id=trace_id,
                    degraded_mode=degraded,
                    uncertainty_level="high",
                    citations=citations,
                    policy_decision=decision,
                    facet_coverage=evidence_bundle.facet_coverage,
                    contradictions=evidence_bundle.contradictions,
                    evidence_ledger=evidence_bundle.evidence_ledger,
                    sufficiency_decision=sufficiency_decision,
                    facet_matrix=facet_matrix,
                    contradiction_ledger=contradiction_ledger,
                    retrieval_traces=trace.retrieval_traces,
                    rerank_traces=trace.rerank_traces,
                )

            answer, provenance = await self.synthesis_agent.synthesize(
                query,
                evidence_bundle,
                facet_coverage=evidence_bundle.facet_coverage,
                safety_notes=[n for n in [decision.reason if decision else None] if n],
            )
            citations = self._citations(evidence_bundle)
            trace.answer_provenance = provenance
            trace.completed_at = datetime.utcnow()
            trace.final_answer = answer
            trace.citations = citations
            await self.trace_repo.create(trace, org_id)
            return ChatResponse(
                answer=answer,
                evidence_bundle=evidence_bundle,
                safety_notes=None,
                trace_id=trace_id,
                degraded_mode=degraded,
                uncertainty_level="medium",
                citations=citations,
                policy_decision=decision,
                facet_coverage=evidence_bundle.facet_coverage,
                contradictions=evidence_bundle.contradictions,
                evidence_ledger=evidence_bundle.evidence_ledger,
                sufficiency_decision=sufficiency_decision,
                facet_matrix=facet_matrix,
                contradiction_ledger=contradiction_ledger,
                answer_provenance=provenance,
                retrieval_traces=trace.retrieval_traces,
                rerank_traces=trace.rerank_traces,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Orchestration failed: %s", exc, exc_info=True)
            degraded["error"] = True
            trace.completed_at = datetime.utcnow()
            trace.degraded_mode = degraded
            await self.trace_repo.create(trace, org_id)
            return ChatResponse(
                answer="I encountered an error processing your query. Please try again or contact support.",
                evidence_bundle=EvidenceBundle(
                    passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0, safety_count=0
                ),
                trace_id=trace_id,
                degraded_mode=degraded,
                citations=[],
                uncertainty_level="high",
            )

    async def _patient_query(
        self,
        query: str,
        org_id: str,
        patient_id: Optional[str],
        constraints: Dict[str, Any],
        trace: AuditTrace,
    ) -> str:
        if not patient_id or constraints.get("disable_patient_retrieval_context"):
            return query
        try:
            chunks = await self.chunk_repo.get_by_patient_id(patient_id, org_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Patient retrieval-context lookup failed: %s", exc)
            return query
        emr_chunks = [
            c for c in chunks
            if str(c.get("source_type", "")).upper() == SourceType.EMR.value
            and (c.get("text") or c.get("content"))
        ]
        if not emr_chunks:
            return query
        text = " ".join(" ".join(str(c.get("text") or c.get("content") or "").split()) for c in emr_chunks[:3])
        synopsis = " ".join(text.split()[:140])
        if not synopsis:
            return query
        trace.messages.append(
            AgentMessage(
                role="system",
                agent_id="retrieval",
                model_endpoint="patient_context_lookup",
                content=f"Built retrieval-only patient synopsis from {len(emr_chunks)} EMR chunks.",
            )
        )
        return f"{query}\n\nPatient retrieval context:\n{redact_phi_text(synopsis)}"

    async def _dispatch_agents(
        self,
        query: str,
        candidates: List,
        facets,
        patient_id: Optional[str],
        routed: Optional[str],
        missing: List[str],
        contradiction_review: bool,
        trace: AuditTrace,
    ):
        batches = []
        # Always run quality on candidates; route specialists by missing facets / hints
        always = {"quality"}
        targets = set(always)
        if routed:
            targets.add(routed)
        if "patient_applicability" in missing or routed == "emr":
            targets.add("emr")
        if "guideline_concordance" in missing or routed == "guideline":
            targets.add("guideline")
        if "safety_contraindications" in missing or routed == "safety":
            targets.add("safety")
        if "evidence_quality" in missing or routed == "quality":
            targets.add("quality")
        if contradiction_review or routed == "critic":
            targets.add("critic")
        # First iteration: explore all specialists lightly when no route yet
        if not routed and not missing:
            targets.update({"emr", "guideline", "safety", "quality", "critic"})

        async def _run(name: str, coro):
            batch = await coro
            batches.append(batch)
            trace.tool_calls.append(
                ToolCall(
                    tool_name=f"agent.{name}",
                    parameters={"query": query, "passage_count": len(candidates)},
                    result={"claims": len(batch.claims), "degraded": batch.degraded, "error": batch.error},
                )
            )
            if batch.degraded:
                trace.degraded_mode[f"agent_{name}"] = True
            trace.messages.append(
                AgentMessage(
                    role="assistant",
                    agent_id=name,
                    content=f"{name} emitted {len(batch.claims)} claims",
                )
            )

        if "emr" in targets:
            await _run("emr", self.emr_agent.explore(query, candidates, facets, patient_id))
        if "guideline" in targets:
            await _run("guideline", self.guideline_agent.explore(query, candidates, facets))
        if "safety" in targets:
            await _run("safety", self.safety_agent.explore(query, candidates, facets))
        if "quality" in targets:
            await _run("quality", self.quality_agent.explore(query, candidates, facets))
        if "critic" in targets:
            await _run("critic", self.critic_agent.explore(query, candidates, facets))
        return batches

    async def _retrieve_with_sufficiency(
        self,
        query: str,
        query_spec: QuerySpec,
        org_id: str,
        patient_id: Optional[str],
        constraints: Dict[str, Any],
        trace: AuditTrace,
    ):
        retrieval_query = await self._patient_query(query, org_id, patient_id, constraints, trace)
        embeddings = await self.embedding_client.embed([retrieval_query])
        query_embedding = embeddings[0] if embeddings else None
        if query_embedding is None:
            empty = EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0)
            return empty, None

        source_type_filter = SourceType.CPG if constraints.get("guideline_only") else None
        facets = query_spec.facets
        iteration = 0
        all_candidates = []
        hints = None
        selected = []
        agent_weights = self.weights.weights()
        final_check = None

        while iteration < settings.MAX_RETRIEVE_LOOPS:
            candidates, retrieval_trace = await self.retriever.retrieve(
                query=retrieval_query,
                query_embedding=query_embedding,
                org_id=org_id,
                source_type_filter=source_type_filter,
                patient_id=patient_id,
                hints=hints,
                constraints=constraints,
            )
            retrieval_trace.iteration = iteration
            trace.retrieval_traces.append(retrieval_trace)
            trace.tool_calls.append(
                ToolCall(
                    tool_name="retrieval.hybrid",
                    parameters={"iteration": iteration, "hints": hints or {}},
                    result={"union_count": retrieval_trace.union_count},
                )
            )

            candidates, rerank_trace = await self.retriever.rerank(retrieval_query, candidates)
            rerank_trace.iteration = iteration
            trace.rerank_traces.append(rerank_trace)
            trace.tool_calls.append(
                ToolCall(
                    tool_name="retrieval.rerank",
                    parameters={"iteration": iteration},
                    result={"calibration_version": rerank_trace.calibration_version, "n": len(candidates)},
                )
            )

            # Merge candidate pool
            by_id = {c.chunk_id: c for c in all_candidates}
            for c in candidates:
                if c.chunk_id not in by_id:
                    by_id[c.chunk_id] = c
                else:
                    existing = by_id[c.chunk_id]
                    existing.dense_score = existing.dense_score or c.dense_score
                    existing.lexical_score = existing.lexical_score or c.lexical_score
                    existing.rerank_score = c.rerank_score or existing.rerank_score
                    existing.calibrated_score = c.calibrated_score or existing.calibrated_score
            all_candidates = list(by_id.values())

            # Agent exploration inside the loop (paper MAC)
            routed = (hints or {}).get("routed_agent")
            missing = (hints or {}).get("missing_facets") or []
            batches = await self._dispatch_agents(
                query=query,
                candidates=all_candidates[:40],
                facets=facets,
                patient_id=patient_id,
                routed=routed,
                missing=missing,
                contradiction_review=bool((hints or {}).get("contradiction_review")),
                trace=trace,
            )

            ledger = build_retrieval_ledger(all_candidates, facets)
            ledger = merge_agent_claims(ledger, batches, all_candidates)
            selected = self.retriever.fuse_and_select(all_candidates, facets=facets, agent_weights=agent_weights)
            selected = pack_bundle(selected, facets=facets)
            selected_ledger = [e for e in ledger if e.chunk_id in {p.chunk_id for p in selected}]
            # Keep full ledger claims for gate but evaluate selected bundle coverage
            check = self.gate.check(
                selected_ledger or ledger,
                facets,
                selected,
                iteration=iteration,
                constraints=constraints,
                query_spec=query_spec,
            )
            # Prefer contradictions from full ledger
            contradictions = detect_contradictions(ledger)
            if check.policy_decision:
                check.policy_decision.contradictions = contradictions
                check.contradiction_count = len(contradictions)
                check.policy_decision.unresolved_critical_conflicts = any(
                    not p.resolved and p.severity == "high" for p in contradictions
                )
            final_check = check
            trace.sufficiency_checks.append(check)
            if check.policy_decision:
                trace.policy_decisions.append(check.policy_decision)

            if check.passed:
                break
            if not self.gate.should_retrieve_more(check):
                break
            hints = check.policy_decision.retrieval_hints if check.policy_decision else {"increase_k": True}
            iteration += 1

        decision = self.gate.last_decision
        coverage = self.gate.last_coverage
        contradictions = self.gate.last_contradictions
        selected_ledger = build_retrieval_ledger(selected, facets)
        # Re-merge agent claims that belong to selected
        # (agent claims already on passages via agent_scores; rebuild from last ledger entries)
        if final_check and final_check.policy_decision:
            decision = final_check.policy_decision
            coverage = decision.facet_coverage
            contradictions = decision.contradictions

        if selected:
            for t in trace.rerank_traces[-1:]:
                t.selected_chunk_ids = [p.chunk_id for p in selected]
                t.rejected_chunk_ids = [
                    c.chunk_id for c in all_candidates if c.chunk_id not in {p.chunk_id for p in selected}
                ][:50]

        bundle = self.retriever.build_bundle(
            selected,
            facet_coverage=coverage,
            evidence_ledger=selected_ledger,
            contradictions=contradictions,
            policy_decision=decision,
        )
        return bundle, final_check

    def _citations(self, evidence_bundle: EvidenceBundle) -> List[Dict[str, Any]]:
        facets_by_chunk = {entry.chunk_id: entry.facets for entry in evidence_bundle.evidence_ledger}
        return [build_citation(passage, facets_by_chunk.get(passage.chunk_id, [])) for passage in evidence_bundle.passages]
