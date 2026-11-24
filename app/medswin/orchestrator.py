"""MedSwin orchestrator: MAC loop with evidence-sufficiency gate."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
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
from app.medswin.ledger import (
    adjudicate_contradictions,
    apply_claim_alignment,
    build_retrieval_ledger,
    detect_contradictions,
    filter_ledger,
    merge_agent_claims,
)
from app.retrieval.hints import expand_query
from app.medswin.normalize import QueryNormalizer
from app.medswin.packing import pack_bundle
from app.repositories.chunks import ChunkRepository
from app.repositories.sessions import SessionRepository
from app.repositories.traces import TraceRepository
from app.retrieval.hybrid import HybridRetriever
from app.schemas.enums import ClinicalScope, PipelineId, PolicyAction, SourceType
from app.schemas.evidence import AnswerArtifact, CandidatePassage, ContradictionLedger, EvidenceBundle, PolicyDecision, QuerySpec, RankedEvidenceItem
from app.schemas.facets import FacetMatrix
from app.schemas.sessions import Session
from app.schemas.traces import AgentMessage, AuditTrace, ChatResponse, ToolCall
from app.scoring.calibrate import get_calibration_store
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.llm import LLMClient
from app.services.adapters.reranker import RerankerClient
from app.services.medswin.governance import build_citation, redact_phi_text

logger = logging.getLogger(__name__)


def merge_candidate_scores(existing: CandidatePassage, incoming: CandidatePassage) -> None:
    """Fill missing retrieval scores. A later zero calibration replaces the old score."""
    if existing.dense_score is None and incoming.dense_score is not None:
        existing.dense_score = incoming.dense_score
    if existing.lexical_score is None and incoming.lexical_score is not None:
        existing.lexical_score = incoming.lexical_score
    if incoming.rerank_score is not None:
        existing.rerank_score = incoming.rerank_score
    if incoming.calibrated_score is not None:
        existing.calibrated_score = incoming.calibrated_score


def _generator_profile(client: LLMClient) -> str:
    return "cloud" if client.backend == "foundry" else "medswin" if client.backend == "medswin_local" else client.backend


@dataclass(frozen=True)
class PipelinePolicy:
    """Immutable behavior contract for the non-naive pipelines."""

    pipeline: PipelineId
    enable_query_normalization: bool
    enable_hybrid_retrieval: bool
    enable_reranker: bool
    enable_mac: bool
    enable_gate: bool
    enable_retrieve_more: bool
    enable_fusion: bool
    ranking_depth: int
    generation_token_budget: int


PIPELINE_POLICIES = {
    PipelineId.TWO_STAGE: PipelinePolicy(
        PipelineId.TWO_STAGE, True, True, True, False, True, True, True, 20, settings.TOKEN_BUDGET_B
    ),
    PipelineId.FULL: PipelinePolicy(
        PipelineId.FULL, True, True, True, True, True, True, True, 20, settings.TOKEN_BUDGET_B
    ),
}


async def _resume_or_create_session(repo, session_id: str, user_id: str, org_id: str) -> str:
    """Keep a session only when this user already owns it. Otherwise mint a new id."""
    updated = await repo.update_last_active(session_id, org_id, user_id=user_id)
    if updated:
        return session_id
    getter = getattr(repo, "get_by_id", None)
    existing = await getter(session_id, org_id) if getter else None
    owner = existing.get("user_id") if isinstance(existing, dict) else None
    if existing is not None and owner == user_id:
        return session_id
    if existing is not None:
        session_id = str(uuid.uuid4())
    await repo.create(Session(session_id=session_id, user_id=user_id, org_id=org_id), org_id)
    return session_id


def _hint_signature(hints: Optional[Dict[str, Any]]) -> tuple:
    hints = hints or {}
    missing = hints.get("missing_facets") or []
    return (
        tuple(missing),
        hints.get("focus_source"),
        bool(hints.get("exclusive_source")),
        bool(hints.get("contradiction_review")),
        bool(hints.get("safety_search")),
    )


def retrieve_stall_action(
    *,
    iteration: int,
    max_loops: int,
    hints: Optional[Dict[str, Any]],
    next_hints: Optional[Dict[str, Any]],
    wants_more: bool,
) -> str:
    """Continue only when the next retrieve would use a different hint."""
    if wants_more and iteration + 1 < max_loops and _hint_signature(next_hints) != _hint_signature(hints):
        return "continue"
    return "abstain"


def _source_filter_for_hints(
    pinned: Optional[SourceType],
    hints: Optional[Dict[str, Any]],
) -> Optional[SourceType]:
    """Let a safety or chart hint override a guideline-only pin for one loop."""
    focus = str((hints or {}).get("focus_source") or "")
    if focus in {item.value for item in SourceType}:
        return SourceType(focus)
    return pinned


class MedSwinOrchestrator:
    """Centralized orchestrator-mediated multi-agent CDSS runtime."""

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
        pipeline: PipelineId = PipelineId.FULL,
    ):
        if pipeline not in PIPELINE_POLICIES:
            raise ValueError(f"MedSwinOrchestrator only supports two-stage/full, got {pipeline}")
        self.policy = PIPELINE_POLICIES[pipeline]
        self.pipeline = pipeline
        cloud_model = settings.CLOUD_MODEL if settings.CLOUD_MODE else None
        self.supervisor = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
        self.emr_llm = self.guideline_llm = self.safety_llm = None
        self.quality_llm = self.critic_llm = None

        self.embedding_client = embedding_client or EmbeddingClient(settings.active_embedding_url())
        self.reranker_client = reranker_client or RerankerClient(settings.active_reranker_url())
        self.retriever = HybridRetriever(reranker_client=self.reranker_client)
        self.gate = EvidenceGate()
        self.normalizer = QueryNormalizer(self.supervisor)
        self.weights = ReliabilityWeights()
        self.calibration = get_calibration_store()

        self.emr_agent = self.guideline_agent = self.safety_agent = None
        self.quality_agent = self.critic_agent = None
        if self.policy.enable_mac:
            self.emr_llm = LLMClient(settings.active_llm_url(settings.AGENT2_URL), model=cloud_model)
            self.guideline_llm = LLMClient(settings.active_llm_url(settings.AGENT3_URL), model=cloud_model)
            self.safety_llm = LLMClient(settings.active_llm_url(settings.AGENT1_URL), model=cloud_model)
            self.quality_llm = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
            self.critic_llm = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
            self.emr_agent = EMRAgent(self.emr_llm)
            self.guideline_agent = GuidelineAgent(self.guideline_llm)
            self.safety_agent = SafetyAgent(self.safety_llm)
            self.quality_agent = QualityAgent(self.quality_llm)
            self.critic_agent = ContradictionAgent(self.critic_llm)
        self.synthesis_agent = SynthesisAgent(self.supervisor)

        self.chunk_repo = ChunkRepository()
        self.session_repo = SessionRepository()
        self.trace_repo = TraceRepository()

    async def _open_session(
        self,
        session_id: Optional[str],
        user_id: str,
        org_id: str,
        degraded: Dict[str, bool],
    ) -> str:
        """Open a session. A database failure does not block the answer."""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
                await self.session_repo.create(
                    Session(session_id=session_id, user_id=user_id, org_id=org_id),
                    org_id,
                )
            else:
                session_id = await _resume_or_create_session(
                    self.session_repo, session_id, user_id, org_id
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Session persistence failed: %s", exc)
            degraded["session"] = True
            session_id = session_id or str(uuid.uuid4())
        return session_id

    async def _persist_trace(self, trace: AuditTrace, org_id: str, degraded: Dict[str, bool]) -> None:
        """Store the audit trace. A database failure does not replace the answer."""
        try:
            await self.trace_repo.create(trace, org_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Trace persistence failed: %s", exc)
            degraded["trace"] = True

    async def chat(
        self,
        query: str,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        benchmark: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        degraded: Dict[str, bool] = {}
        if not self.calibration.loaded:
            degraded["calibration"] = True
        session_id = await self._open_session(session_id, user_id, org_id, degraded)

        trace = AuditTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            query=query,
            patient_id=patient_id,
            pipeline=self.pipeline,
        )
        ranked_evidence: List[RankedEvidenceItem] = []

        try:
            normalize_started = time.perf_counter()
            query_spec = await self.normalizer.normalize(query, trace)
            trace.timing_ms["normalize"] = round((time.perf_counter() - normalize_started) * 1000.0, 2)
            facets = self.normalizer.build_facets(query, query_spec, constraints, patient_id)
            query_spec.facets = facets
            if constraints and constraints.get("clinical_scope"):
                try:
                    query_spec.clinical_scope = ClinicalScope(constraints["clinical_scope"])
                except ValueError:
                    query_spec.clinical_scope = ClinicalScope.CLINICIAN_CDS

            retrieve_started = time.perf_counter()
            evidence_bundle, final_check = await self._retrieve_with_sufficiency(
                query=query,
                query_spec=query_spec,
                org_id=org_id,
                patient_id=patient_id,
                constraints=constraints or {},
                trace=trace,
                ranked_out=ranked_evidence,
            )
            trace.timing_ms["retrieve_gate"] = round((time.perf_counter() - retrieve_started) * 1000.0, 2)

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
            if decision is None or (decision.passed and not evidence_bundle.passages):
                decision = PolicyDecision(
                    passed=False,
                    action=PolicyAction.INSUFFICIENT_EVIDENCE,
                    reason="Evidence state is empty or missing a sufficiency decision.",
                    missing_facets=[facet.name for facet in query_spec.facets if facet.required],
                )
                evidence_bundle.policy_decision = decision
            sufficiency_decision = self.gate.to_sufficiency_decision(decision)
            trace.facet_matrix = facet_matrix
            trace.contradiction_ledger = contradiction_ledger
            trace.sufficiency_decision = sufficiency_decision
            trace.evidence_bundle = evidence_bundle
            trace.facet_coverage = evidence_bundle.facet_coverage
            trace.contradictions = evidence_bundle.contradictions
            trace.evidence_ledger = evidence_bundle.evidence_ledger
            if any(item.error for item in trace.retrieval_traces):
                degraded["retrieval"] = True
            trace.degraded_mode = degraded

            if not decision.passed:
                answer = insufficient_answer(query, decision, evidence_bundle.facet_coverage)
                citations = self._citations(evidence_bundle)
                trace.completed_at = datetime.utcnow()
                trace.final_answer = answer
                trace.citations = citations
                trace.timing_ms["total"] = round((time.perf_counter() - started) * 1000.0, 2)
                await self._persist_trace(trace, org_id, degraded)
                return ChatResponse(
                    answer=answer,
                    answer_artifact=AnswerArtifact(content="", rendered_text=answer, exact_answer=None, ideal_answer="" if benchmark else None, cited_chunk_ids=[c["chunk_id"] for c in citations], abstained=True),
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
                    pipeline=self.pipeline,
                    generator_profile=_generator_profile(self.supervisor),
                    ranked_evidence=ranked_evidence,
                    timing_ms=trace.timing_ms,
                )

            synthesis_started = time.perf_counter()
            answer, provenance, answer_artifact = await self.synthesis_agent.synthesize_with_artifact(
                query,
                evidence_bundle,
                facet_coverage=evidence_bundle.facet_coverage,
                safety_notes=[n for n in [decision.reason if decision else None] if n],
                benchmark=bool(benchmark),
            )
            trace.timing_ms["synthesis"] = round((time.perf_counter() - synthesis_started) * 1000.0, 2)
            citations = self._citations(evidence_bundle)
            trace.answer_provenance = provenance
            trace.completed_at = datetime.utcnow()
            trace.final_answer = answer
            trace.citations = citations
            trace.timing_ms["total"] = round((time.perf_counter() - started) * 1000.0, 2)
            await self._persist_trace(trace, org_id, degraded)
            return ChatResponse(
                answer=answer,
                answer_artifact=answer_artifact,
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
                pipeline=self.pipeline,
                generator_profile=_generator_profile(self.supervisor),
                ranked_evidence=ranked_evidence,
                timing_ms=trace.timing_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Orchestration failed: %s", exc, exc_info=True)
            degraded["error"] = True
            trace.completed_at = datetime.utcnow()
            trace.degraded_mode = degraded
            trace.timing_ms["total"] = round((time.perf_counter() - started) * 1000.0, 2)
            await self._persist_trace(trace, org_id, degraded)
            return ChatResponse(
                answer="I encountered an error processing your query. Please try again or contact support.",
                answer_artifact=AnswerArtifact(
                    content="",
                    rendered_text="I encountered an error processing your query. Please try again or contact support.",
                    abstained=True,
                ),
                evidence_bundle=EvidenceBundle(
                    passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0, safety_count=0
                ),
                trace_id=trace_id,
                degraded_mode=degraded,
                citations=[],
                uncertainty_level="high",
                pipeline=self.pipeline,
                generator_profile=_generator_profile(self.supervisor),
                ranked_evidence=ranked_evidence,
                timing_ms=trace.timing_ms,
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
        if not self.policy.enable_mac:
            trace.tool_calls.append(ToolCall(tool_name="pipeline.mac", parameters={}, result={"status": "skipped_by_pipeline"}))
            return []
        if not candidates:
            trace.tool_calls.append(
                ToolCall(
                    tool_name="pipeline.mac",
                    parameters={"passage_count": 0},
                    result={"status": "skipped_no_candidates"},
                )
            )
            return []
        batches = []
        always = {"quality"}
        targets = set(always)
        if isinstance(routed, list):
            targets.update(routed)
        elif routed:
            targets.add(routed)
        if "patient_applicability" in missing:
            targets.add("emr")
        if "guideline_concordance" in missing:
            targets.add("guideline")
        if "safety_contraindications" in missing:
            targets.add("safety")
        if "evidence_quality" in missing:
            targets.add("quality")
        if contradiction_review:
            targets.add("critic")
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

        if "emr" in targets and self.emr_agent:
            await _run("emr", self.emr_agent.explore(query, candidates, facets, patient_id))
        if "guideline" in targets and self.guideline_agent:
            await _run("guideline", self.guideline_agent.explore(query, candidates, facets))
        if "safety" in targets and self.safety_agent:
            await _run("safety", self.safety_agent.explore(query, candidates, facets))
        if "quality" in targets and self.quality_agent:
            await _run("quality", self.quality_agent.explore(query, candidates, facets))
        if "critic" in targets and self.critic_agent:
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
        ranked_out: Optional[List[RankedEvidenceItem]] = None,
    ):
        retrieval_query = await self._patient_query(query, org_id, patient_id, constraints, trace)
        embeddings = await self.embedding_client.embed([retrieval_query])
        query_embedding = embeddings[0] if embeddings else None
        if query_embedding is None:
            empty = EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0)
            return empty, None

        source_type_filter = SourceType.CPG if constraints.get("guideline_only") else None
        facets = query_spec.facets or QueryNormalizer.build_facets(query, query_spec, constraints, patient_id)
        if not query_spec.facets:
            query_spec.facets = facets
        iteration = 0
        all_candidates = []
        hints = None
        selected = []
        selected_ledger = []
        agent_weights = self.weights.weights()
        final_check = None
        last_embedded = retrieval_query

        while iteration < settings.MAX_RETRIEVE_LOOPS:
            loop_query = expand_query(retrieval_query, hints)
            if loop_query != last_embedded:
                expanded_embeddings = await self.embedding_client.embed([loop_query])
                if expanded_embeddings:
                    query_embedding = expanded_embeddings[0]
                    last_embedded = loop_query
            loop_filter = _source_filter_for_hints(source_type_filter, hints)
            candidates, retrieval_trace = await self.retriever.retrieve(
                query=loop_query,
                query_embedding=query_embedding,
                org_id=org_id,
                source_type_filter=loop_filter,
                patient_id=patient_id,
                hints=hints,
                constraints=constraints,
            )
            retrieval_trace.iteration = iteration
            trace.retrieval_traces.append(retrieval_trace)
            if retrieval_trace.error:
                check = self.gate.check(
                    [],
                    facets,
                    [],
                    iteration=iteration,
                    constraints=constraints,
                    query_spec=query_spec,
                    stop_retrieval=True,
                )
                if check.policy_decision:
                    check.policy_decision.reason = "Retrieval failed."
                    check.policy_decision.action = PolicyAction.INSUFFICIENT_EVIDENCE
                    check.policy_decision.passed = False
                    check.policy_decision.retrieval_hints = {}
                check.passed = False
                check.action_taken = PolicyAction.INSUFFICIENT_EVIDENCE.value
                final_check = check
                trace.sufficiency_checks.append(check)
                if check.policy_decision:
                    trace.policy_decisions.append(check.policy_decision)
                trace.tool_calls.append(
                    ToolCall(
                        tool_name="retrieval.hybrid",
                        parameters={"iteration": iteration, "hints": hints or {}},
                        result={"union_count": 0, "error": retrieval_trace.error},
                    )
                )
                break
            trace.tool_calls.append(
                ToolCall(
                    tool_name="retrieval.hybrid",
                    parameters={"iteration": iteration, "hints": hints or {}},
                    result={"union_count": retrieval_trace.union_count},
                )
            )

            candidates, rerank_trace = await self.retriever.rerank(loop_query, candidates)
            rerank_trace.iteration = iteration
            trace.rerank_traces.append(rerank_trace)
            trace.tool_calls.append(
                ToolCall(
                    tool_name="retrieval.rerank",
                    parameters={"iteration": iteration},
                    result={"calibration_version": rerank_trace.calibration_version, "n": len(candidates)},
                )
            )

            by_id = {c.chunk_id: c for c in all_candidates}
            prev_ids = set(by_id.keys())
            for c in candidates:
                if c.chunk_id not in by_id:
                    by_id[c.chunk_id] = c
                else:
                    merge_candidate_scores(by_id[c.chunk_id], c)
            all_candidates = list(by_id.values())
            new_ids = set(by_id.keys()) - prev_ids

            if iteration > 0 and not new_ids:
                trace.tool_calls.append(
                    ToolCall(
                        tool_name="pipeline.mac",
                        parameters={"iteration": iteration, "skipped": "no_new_candidates"},
                        result={"count": 0},
                    )
                )
                preview = self.gate.check(
                    selected_ledger,
                    facets,
                    selected,
                    iteration=iteration,
                    constraints=constraints,
                    query_spec=query_spec,
                    contradictions=contradictions if "contradictions" in locals() else [],
                )
                next_hints = (
                    preview.policy_decision.retrieval_hints if preview.policy_decision else {}
                )
                if retrieve_stall_action(
                    iteration=iteration,
                    max_loops=settings.MAX_RETRIEVE_LOOPS,
                    hints=hints,
                    next_hints=next_hints,
                    wants_more=self.policy.enable_retrieve_more and self.gate.should_retrieve_more(preview),
                ) == "continue":
                    final_check = preview
                    trace.sufficiency_checks.append(preview)
                    if preview.policy_decision:
                        trace.policy_decisions.append(preview.policy_decision)
                    hints = next_hints
                    iteration += 1
                    continue
                check = self.gate.check(
                    selected_ledger,
                    facets,
                    selected,
                    iteration=iteration,
                    constraints=constraints,
                    query_spec=query_spec,
                    contradictions=contradictions if "contradictions" in locals() else [],
                    stop_retrieval=True,
                )
                final_check = check
                trace.sufficiency_checks.append(check)
                if check.policy_decision:
                    trace.policy_decisions.append(check.policy_decision)
                break

            routed = (hints or {}).get("routed_agents") or (hints or {}).get("routed_agent")
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

            ranked_candidates = self.retriever.rank(all_candidates, facets=facets)
            ledger = build_retrieval_ledger(ranked_candidates, facets)
            if self.policy.enable_mac:
                ledger = merge_agent_claims(ledger, batches, ranked_candidates)
                apply_claim_alignment(ranked_candidates, batches, facets)
            budget = self.policy.generation_token_budget
            selected = self.retriever.select(
                ranked_candidates,
                facets=facets,
                agent_weights=agent_weights,
                token_budget=budget,
            )
            selected = pack_bundle(selected, facets=facets, token_budget=budget)
            selected_ids = {p.chunk_id for p in selected}
            selected_ledger = filter_ledger(ledger, selected_ids)
            detected_contradictions = detect_contradictions(selected_ledger)
            contradictions = (
                adjudicate_contradictions(detected_contradictions, batches)
                if self.policy.enable_mac
                else detected_contradictions
            )
            check = self.gate.check(
                selected_ledger,
                facets,
                selected,
                iteration=iteration,
                constraints=constraints,
                query_spec=query_spec,
                contradictions=contradictions,
            )
            final_check = check
            trace.sufficiency_checks.append(check)
            if check.policy_decision:
                trace.policy_decisions.append(check.policy_decision)

            if check.passed:
                break
            if not self.policy.enable_retrieve_more or not self.gate.should_retrieve_more(check):
                break
            hints = check.policy_decision.retrieval_hints if check.policy_decision else {"increase_k": True}
            iteration += 1

        decision = None
        coverage = []
        contradictions = []
        if final_check and final_check.policy_decision:
            decision = final_check.policy_decision
            coverage = decision.facet_coverage
            contradictions = decision.contradictions

        # ``ranked_candidates`` is the final post-rerank/fusion state, before
        # MMR/token packing. It is the evaluator's observation point.
        ranked = self._document_rank(
            ranked_candidates if "ranked_candidates" in locals() else all_candidates,
            depth=self.policy.ranking_depth,
        )
        if ranked_out is not None:
            ranked_out[:] = ranked
        if selected:
            for trace_row in trace.rerank_traces[-1:]:
                trace_row.selected_chunk_ids = [p.chunk_id for p in selected]
                trace_row.rejected_chunk_ids = [
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

    @staticmethod
    def _document_rank(candidates: List, depth: int = 20) -> List[RankedEvidenceItem]:
        """Collapse final pre-pack chunk scores into a stable document ranking."""
        by_doc = {}
        for candidate in candidates:
            doc_id = str(candidate.doc_id or candidate.chunk_id)
            score = float(candidate.fusion_score if candidate.fusion_score is not None else (
                candidate.calibrated_score if candidate.calibrated_score is not None else (
                    candidate.rerank_score if candidate.rerank_score is not None else (candidate.dense_score or 0.0)
                )
            ))
            current = by_doc.get(doc_id)
            if current is None or score > current[0] or (score == current[0] and candidate.chunk_id < current[1].chunk_id):
                by_doc[doc_id] = (score, candidate)
        ordered = sorted(by_doc.items(), key=lambda item: (-item[1][0], item[0]))[:depth]
        return [
            RankedEvidenceItem(
                doc_id=doc_id,
                winning_chunk_id=candidate.chunk_id,
                rank=index,
                score=score,
                dense_score=candidate.dense_score,
                lexical_score=candidate.lexical_score,
                rerank_score=candidate.rerank_score,
                fusion_score=candidate.fusion_score,
                source_type=candidate.source_type,
            )
            for index, (doc_id, (score, candidate)) in enumerate(ordered, start=1)
        ]

    def _render_final_answer(self, answer_data: Dict[str, Any], evidence_bundle: EvidenceBundle) -> str:
        """Backward-compatible grounded renderer used by safety/unit callers.

        Citation filtering lives in ``SynthesisAgent._render``. Delegating here
        keeps one implementation: fabricated chunk IDs are rejected against the
        accepted EvidenceBundle rather than reimplementing grounding logic in
        the orchestrator.
        """
        answer, _provenance = self.synthesis_agent._render(answer_data, evidence_bundle)
        return answer

    def _citations(self, evidence_bundle: EvidenceBundle) -> List[Dict[str, Any]]:
        facets_by_chunk = {entry.chunk_id: entry.facets for entry in evidence_bundle.evidence_ledger}
        return [build_citation(passage, facets_by_chunk.get(passage.chunk_id, [])) for passage in evidence_bundle.passages]

    async def close(self) -> None:
        """Close each owned HTTP client exactly once."""
        clients = [
            self.supervisor,
            self.emr_llm,
            self.guideline_llm,
            self.safety_llm,
            self.quality_llm,
            self.critic_llm,
            self.embedding_client,
            self.reranker_client,
        ]
        seen = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if close is not None:
                await close()
