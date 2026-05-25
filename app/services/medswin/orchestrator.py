"""MedSwin orchestrator with supervisor + specialist agents."""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.config import settings
from app.services.adapters.llm import LLMClient
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient
from app.services.medswin.retrieval import RetrievalPipeline
from app.services.medswin.policy import EvidenceSufficiencyPolicy
from app.services.prompts import answer as answer_prompt
from app.services.prompts import emr as emr_prompt
from app.services.prompts import guideline as guideline_prompt
from app.services.prompts import query as query_prompt
from app.services.prompts import safety as safety_prompt
from app.services.prompts.structured import extract_json_object
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.sessions import SessionRepository
from app.repositories.traces import TraceRepository
from app.models.medswin import (
    QuerySpec,
    CandidatePassage,
    EvidenceBundle,
    EMRSummary,
    GuidelineSummary,
    SafetyReport,
    ChatResponse,
    AuditTrace,
    AgentMessage,
    ToolCall,
    SufficiencyCheck,
    SourceType,
    PolicyAction,
    ClinicalScope
)
from app.services.medswin.governance import build_citation, ensure_cds_language, redact_phi_text

logger = logging.getLogger(__name__)


class MedSwinOrchestrator:
    """Orchestrator for MedSwin multi-agent conversation."""
    
    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None
    ):
        """Initialize orchestrator.
        
        Args:
            embedding_client: Optional embedding client
            reranker_client: Optional reranker client
        """
        # Initialize clients
        cloud_model = settings.CLOUD_MODEL if settings.CLOUD_MODE else None
        self.supervisor_client = LLMClient(settings.active_llm_url(settings.SUPERVISOR_URL), model=cloud_model)
        self.agent1_client = LLMClient(settings.active_llm_url(settings.AGENT1_URL), model=cloud_model)
        self.agent2_client = LLMClient(settings.active_llm_url(settings.AGENT2_URL), model=cloud_model)
        self.agent3_client = LLMClient(settings.active_llm_url(settings.AGENT3_URL), model=cloud_model)
        
        self.embedding_client = embedding_client or EmbeddingClient(settings.active_embedding_url())
        self.reranker_client = reranker_client or RerankerClient(settings.active_reranker_url())
        
        # Initialize services
        self.retrieval_pipeline = RetrievalPipeline(
            embedding_client=self.embedding_client,
            reranker_client=self.reranker_client
        )
        self.sufficiency_policy = EvidenceSufficiencyPolicy()
        
        # Initialize repositories
        self.chunk_repo = ChunkRepository()
        self.doc_repo = DocumentRepository()
        self.session_repo = SessionRepository()
        self.trace_repo = TraceRepository()
    
    async def chat(
        self,
        query: str,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Process a chat query through the MedSwin pipeline.
        
        Args:
            query: User query
            user_id: User ID
            org_id: Organization ID
            session_id: Optional session ID (creates new if missing)
            patient_id: Optional patient ID
            constraints: Optional constraints (guideline-only, timeframe, etc.)
            
        Returns:
            ChatResponse with answer, evidence bundle, and trace
        """
        trace_id = str(uuid.uuid4())
        
        # Create or get session
        if not session_id:
            session_id = str(uuid.uuid4())
            from app.models.medswin import Session
            session = Session(session_id=session_id, user_id=user_id, org_id=org_id)
            await self.session_repo.create(session, org_id)
        else:
            await self.session_repo.update_last_active(session_id, org_id)
        
        # Initialize trace
        trace = AuditTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            query=query,
            patient_id=patient_id
        )
        
        degraded_mode = {}
        
        try:
            # Step 1: Normalize query (supervisor)
            query_spec = await self._normalize_query(query, trace)
            
            # Step 2: Retrieve candidates with sufficiency loop
            evidence_bundle = await self._retrieve_with_sufficiency(
                query,
                query_spec,
                org_id,
                patient_id,
                constraints,
                trace
            )

            if evidence_bundle.policy_decision and not evidence_bundle.policy_decision.passed:
                answer = self._insufficient_evidence_answer(query, evidence_bundle.policy_decision)
                citations = self._build_citations(evidence_bundle)
                trace.completed_at = datetime.utcnow()
                trace.final_answer = answer
                trace.citations = citations
                await self.trace_repo.create(trace, org_id)
                return ChatResponse(
                    answer=answer,
                    evidence_bundle=evidence_bundle,
                    safety_notes=evidence_bundle.policy_decision.reason,
                    trace_id=trace_id,
                    degraded_mode=degraded_mode,
                    uncertainty_level="high",
                    citations=citations,
                    policy_decision=evidence_bundle.policy_decision,
                    facet_coverage=evidence_bundle.facet_coverage,
                    contradictions=evidence_bundle.contradictions,
                    evidence_ledger=evidence_bundle.evidence_ledger,
                )
            
            # Step 3: EMR Summary (Agent 2)
            emr_summary = await self._summarize_emr(
                evidence_bundle,
                patient_id,
                org_id,
                trace
            )
            
            # Step 4: Guideline Synthesis (Agent 3)
            guideline_summary = await self._synthesize_guidelines(
                evidence_bundle,
                trace
            )
            
            # Step 5: Safety Critique
            safety_report = await self._safety_critique(
                evidence_bundle,
                emr_summary,
                guideline_summary,
                query,
                trace
            )
            
            # Step 6: Final answer (supervisor)
            answer = await self._generate_final_answer(
                query,
                evidence_bundle,
                emr_summary,
                guideline_summary,
                safety_report,
                trace
            )
            
            # Build citations
            citations = self._build_citations(evidence_bundle)
            
            trace.completed_at = datetime.utcnow()
            trace.final_answer = answer
            trace.citations = citations
            
            # Save trace
            await self.trace_repo.create(trace, org_id)
            
            return ChatResponse(
                answer=answer,
                evidence_bundle=evidence_bundle,
                safety_notes=safety_report.unsafe_suggestions[0] if safety_report.unsafe_suggestions else None,
                trace_id=trace_id,
                degraded_mode=degraded_mode,
                uncertainty_level="high" if safety_report.insufficient_evidence else "medium",
                citations=citations,
                policy_decision=evidence_bundle.policy_decision,
                facet_coverage=evidence_bundle.facet_coverage,
                contradictions=evidence_bundle.contradictions,
                evidence_ledger=evidence_bundle.evidence_ledger,
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            trace.completed_at = datetime.utcnow()
            await self.trace_repo.create(trace, org_id)
            
            return ChatResponse(
                answer=f"I encountered an error processing your query. Please try again or contact support.",
                evidence_bundle=EvidenceBundle(
                    passages=[],
                    total_tokens=0,
                    cpg_count=0,
                    emr_count=0,
                    lit_count=0
                ),
                trace_id=trace_id,
                degraded_mode={"error": True},
                citations=[],
                uncertainty_level="high"
            )
    
    async def _normalize_query(self, query: str, trace: AuditTrace) -> QuerySpec:
        """Normalize query using supervisor."""
        messages = [
            {
                "role": "system",
                "content": query_prompt.SYSTEM
            },
            {
                "role": "user",
                "content": f"Normalize this medical query: {query}"
            }
        ]
        
        try:
            response = await self.supervisor_client.call_llm(
                messages,
                json_schema=query_prompt.SCHEMA
            )
            
            spec_data = self._coerce_query_spec_data(extract_json_object(response["content"]))
            query_spec = QuerySpec(**spec_data)
            
            trace.messages.append(AgentMessage(
                role="assistant",
                agent_id="supervisor",
                model_endpoint=settings.SUPERVISOR_URL,
                content=f"Normalized query: {query_spec.canonical_terms}",
                token_count=response.get("token_count")
            ))
            
            return query_spec
            
        except Exception as e:
            logger.warning(f"Query normalization failed: {e}")
            # Return basic spec
            return QuerySpec(canonical_terms=[query])

    def _coerce_query_spec_data(self, spec_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce model JSON into the strict QuerySpec contract.

        Root Cause vs Logic: Query-normalization LLMs can emit descriptive
        strings where the runtime requires enums/floats. Coercing recoverable
        fields preserves useful normalized terms while preventing benign schema
        drift from disabling retrieval normalization entirely.
        """
        if not isinstance(spec_data, dict):
            return {"canonical_terms": [], "clinical_scope": ClinicalScope.CLINICIAN_CDS.value}

        data = dict(spec_data)
        data.setdefault("canonical_terms", [])
        data.setdefault("abbreviations", {})
        data.setdefault("retrieval_hints", {})

        try:
            data["clinical_scope"] = ClinicalScope(data.get("clinical_scope")).value
        except (TypeError, ValueError):
            data["clinical_scope"] = ClinicalScope.CLINICIAN_CDS.value

        coerced_facets = []
        for facet in data.get("facets") or []:
            if isinstance(facet, str):
                facet = {"name": facet}
            if not isinstance(facet, dict):
                continue
            item = dict(facet)
            item["name"] = str(item.get("name") or "clinical_evidence")
            item["required"] = bool(item.get("required", True))
            item["threshold"] = self._coerce_float(item.get("threshold"), 0.70)
            item["weight"] = self._coerce_float(item.get("weight"), 1.0)
            keywords = item.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            item["keywords"] = [str(keyword) for keyword in keywords if str(keyword).strip()]
            coerced_facets.append(item)
        data["facets"] = coerced_facets
        return data

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    async def _retrieve_with_sufficiency(
        self,
        query: str,
        query_spec: QuerySpec,
        org_id: str,
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
        trace: AuditTrace
    ) -> EvidenceBundle:
        """Retrieve evidence with sufficiency loop."""
        retrieval_query = await self._patient_aware_retrieval_query(query, org_id, patient_id, constraints, trace)
        # Generate query embedding
        embeddings = await self.embedding_client.embed([retrieval_query])
        query_embedding = embeddings[0] if embeddings else None
        
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0)
        
        # Determine source type filter from constraints
        source_type_filter = None
        if constraints and constraints.get("guideline_only"):
            source_type_filter = SourceType.CPG
        
        facets = self.sufficiency_policy.build_facets(query, query_spec, constraints, patient_id)
        query_spec.facets = facets
        if constraints and constraints.get("clinical_scope"):
            try:
                query_spec.clinical_scope = ClinicalScope(constraints["clinical_scope"])
            except ValueError:
                query_spec.clinical_scope = ClinicalScope.CLINICIAN_CDS

        iteration = 0
        all_candidates = []
        hints = None
        selected: List[CandidatePassage] = []
        
        while iteration < settings.MAX_RETRIEVE_LOOPS:
            # Retrieve candidates
            candidates = await self.retrieval_pipeline.retrieve(
                query=retrieval_query,
                query_embedding=query_embedding,
                org_id=org_id,
                source_type_filter=source_type_filter,
                patient_id=patient_id,
                hints=hints,
                constraints=constraints,
            )
            
            # Rerank
            candidates = await self.retrieval_pipeline.rerank(retrieval_query, candidates)
            
            # Compute fusion scores
            candidates = self.retrieval_pipeline.compute_fusion_scores(candidates)
            
            # Merge with previous candidates
            candidate_dict = {c.chunk_id: c for c in all_candidates}
            for c in candidates:
                if c.chunk_id not in candidate_dict:
                    candidate_dict[c.chunk_id] = c
            all_candidates = list(candidate_dict.values())

            for candidate in all_candidates:
                self.sufficiency_policy.score_passage_facets(candidate, facets)

            selected = self.retrieval_pipeline.select_with_mmr(
                all_candidates,
                query_embedding,
                facets=facets
            )

            # Check sufficiency over the selected bundle, while preserving candidate recall.
            check = self.sufficiency_policy.check_sufficiency(
                all_candidates,
                iteration,
                query_spec=query_spec,
                constraints=constraints,
                patient_id=patient_id,
                selected_passages=selected
            )
            trace.sufficiency_checks.append(check)
            if check.policy_decision:
                trace.policy_decisions.append(check.policy_decision)
                trace.facet_coverage = check.policy_decision.facet_coverage
                trace.contradictions = check.policy_decision.contradictions
            
            if check.passed:
                break
            
            if not self.sufficiency_policy.should_retrieve_more(check):
                break
            
            hints = self.sufficiency_policy.get_retrieval_hints(check)
            iteration += 1
        
        if not selected:
            selected = self.retrieval_pipeline.select_with_mmr(
                all_candidates,
                query_embedding,
                facets=facets
            )

        final_check = self.sufficiency_policy.check_sufficiency(
            all_candidates,
            iteration,
            query_spec=query_spec,
            constraints=constraints,
            patient_id=patient_id,
            selected_passages=selected
        )
        if final_check.policy_decision and (
            not trace.policy_decisions or trace.policy_decisions[-1] != final_check.policy_decision
        ):
            trace.policy_decisions.append(final_check.policy_decision)
        trace.evidence_ledger = self.sufficiency_policy.last_evidence_ledger
        trace.facet_coverage = self.sufficiency_policy.last_facet_coverage
        trace.contradictions = self.sufficiency_policy.last_contradictions
        
        # Build evidence bundle
        evidence_bundle = self.retrieval_pipeline.build_evidence_bundle(
            selected,
            facet_coverage=self.sufficiency_policy.last_facet_coverage,
            evidence_ledger=self.sufficiency_policy.last_evidence_ledger,
            contradictions=self.sufficiency_policy.last_contradictions,
            policy_decision=self.sufficiency_policy.last_policy_decision
        )
        trace.evidence_bundle = evidence_bundle
        
        return evidence_bundle

    async def _patient_aware_retrieval_query(
        self,
        query: str,
        org_id: str,
        patient_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
        trace: AuditTrace,
    ) -> str:
        """Build retrieval-only patient context without changing the chat query.

        Motivation vs Logic: TREC CDS relevance depends on the patient note, but
        stuffing the note into `/medswin/chat` would blur prompt augmentation
        with retrieval quality. This helper uses a short EMR synopsis only for
        dense/BM25 retrieval, while the final answer still receives the original
        clinician query and selected evidence.
        """
        if not patient_id or (constraints or {}).get("disable_patient_retrieval_context"):
            return query
        try:
            chunks = await self.chunk_repo.get_by_patient_id(patient_id, org_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Patient retrieval-context lookup failed: %s", exc)
            return query
        emr_chunks = [
            chunk for chunk in chunks
            if str(chunk.get("source_type", "")).upper() == SourceType.EMR.value and (chunk.get("text") or chunk.get("content"))
        ]
        if not emr_chunks:
            return query
        text = " ".join(" ".join(str(chunk.get("text") or chunk.get("content") or "").split()) for chunk in emr_chunks[:3])
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
    
    async def _summarize_emr(
        self,
        evidence_bundle: EvidenceBundle,
        patient_id: Optional[str],
        org_id: str,
        trace: AuditTrace
    ) -> EMRSummary:
        """Summarize EMR using Agent 2."""
        emr_passages = [p for p in evidence_bundle.passages if p.source_type == SourceType.EMR]
        
        if not emr_passages:
            return EMRSummary()
        
        # Build context from EMR passages
        context = "\n\n".join([p.text for p in emr_passages])
        
        messages = [
            {
                "role": "system",
                "content": emr_prompt.SYSTEM
            },
            {
                "role": "user",
                "content": f"Summarize this EMR information:\n\n{context}"
            }
        ]
        
        try:
            response = await self.agent2_client.call_llm(
                messages,
                json_schema=emr_prompt.SCHEMA
            )
            
            summary_data = extract_json_object(response["content"])
            summary_data = self._coerce_emr_summary_data(summary_data)
            emr_summary = EMRSummary(patient_id=patient_id, **summary_data)
            
            trace.messages.append(AgentMessage(
                role="assistant",
                agent_id="agent2",
                model_endpoint=settings.AGENT2_URL,
                content=f"EMR summary: {len(emr_summary.problems)} problems, {len(emr_summary.medications)} medications",
                token_count=response.get("token_count")
            ))
            
            return emr_summary
            
        except Exception as e:
            logger.warning(f"EMR summarization failed: {e}")
            return EMRSummary()

    def _coerce_emr_summary_data(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce common LLM schema drift in EMR summaries.

        Root Cause vs Logic: cloud chat models sometimes emit `timeline` as a
        list of strings even when the schema requests objects. We preserve the
        useful clinical text by wrapping each item in a small event dictionary
        instead of discarding the whole EMR summary.
        """
        data = dict(summary_data or {})
        timeline = data.get("timeline")
        if isinstance(timeline, list):
            coerced = []
            for item in timeline:
                if isinstance(item, dict):
                    coerced.append(item)
                elif item:
                    coerced.append({"event": str(item)})
            data["timeline"] = coerced
        return data
    
    async def _synthesize_guidelines(
        self,
        evidence_bundle: EvidenceBundle,
        trace: AuditTrace
    ) -> GuidelineSummary:
        """Synthesize guidelines using Agent 3."""
        cpg_passages = [p for p in evidence_bundle.passages if p.source_type == SourceType.CPG]
        
        if not cpg_passages:
            return GuidelineSummary()
        
        # Build context from CPG passages
        context = "\n\n".join([p.text for p in cpg_passages])
        
        messages = [
            {
                "role": "system",
                "content": guideline_prompt.SYSTEM
            },
            {
                "role": "user",
                "content": f"Synthesize guidelines from:\n\n{context}"
            }
        ]
        
        try:
            response = await self.agent3_client.call_llm(
                messages,
                json_schema=guideline_prompt.SCHEMA
            )
            
            summary_data = extract_json_object(response["content"])
            guideline_summary = GuidelineSummary(**summary_data)
            
            trace.messages.append(AgentMessage(
                role="assistant",
                agent_id="agent3",
                model_endpoint=settings.AGENT3_URL,
                content=f"Guideline synthesis: {len(guideline_summary.recommendations)} recommendations",
                token_count=response.get("token_count")
            ))
            
            return guideline_summary
            
        except Exception as e:
            logger.warning(f"Guideline synthesis failed: {e}")
            return GuidelineSummary()
    
    async def _safety_critique(
        self,
        evidence_bundle: EvidenceBundle,
        emr_summary: EMRSummary,
        guideline_summary: GuidelineSummary,
        query: str,
        trace: AuditTrace
    ) -> SafetyReport:
        """Perform safety critique (supervisor)."""
        messages = [
            {
                "role": "system",
                "content": safety_prompt.SYSTEM
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nEMR Summary: {emr_summary.model_dump_json()}\n\nGuidelines: {guideline_summary.model_dump_json()}\n\nEvidence: {len(evidence_bundle.passages)} passages"
            }
        ]
        
        try:
            response = await self.supervisor_client.call_llm(
                messages,
                json_schema=safety_prompt.SCHEMA
            )
            
            report_data = extract_json_object(response["content"])
            safety_report = SafetyReport(**report_data)
            
            trace.messages.append(AgentMessage(
                role="assistant",
                agent_id="supervisor",
                model_endpoint=settings.SUPERVISOR_URL,
                content=f"Safety critique: {len(safety_report.unsafe_suggestions)} unsafe suggestions",
                token_count=response.get("token_count")
            ))
            
            return safety_report
            
        except Exception as e:
            logger.warning(f"Safety critique failed: {e}")
            return SafetyReport()
    
    async def _generate_final_answer(
        self,
        query: str,
        evidence_bundle: EvidenceBundle,
        emr_summary: EMRSummary,
        guideline_summary: GuidelineSummary,
        safety_report: SafetyReport,
        trace: AuditTrace
    ) -> str:
        """Generate final answer using supervisor."""
        # Build evidence context
        evidence_text = "\n\n".join([
            f"[{p.chunk_id}] {p.text}" for p in evidence_bundle.passages
        ])
        
        messages = [
            {
                "role": "system",
                "content": answer_prompt.SYSTEM
            },
            {
                "role": "user",
                "content": f"""Query: {query}

EMR Summary:
{emr_summary.model_dump_json()}

Guidelines:
{guideline_summary.model_dump_json()}

Evidence:
{evidence_text}

Safety Notes:
{safety_report.model_dump_json()}

Provide a clinician decision-support answer with:
1. Answer to the query
2. Evidence used section (reference chunk_ids)
3. Explicit uncertainty language if evidence is insufficient
4. Contraindications/risks if present
5. Recommended next steps for clinician review
Do not present autonomous diagnosis or treatment orders."""
            }
        ]
        
        try:
            response = await self.supervisor_client.call_llm(messages, json_schema=answer_prompt.SCHEMA)
            answer_data = extract_json_object(response["content"])
            answer = ensure_cds_language(self._render_final_answer(answer_data, evidence_bundle))
            
            trace.messages.append(AgentMessage(
                role="assistant",
                agent_id="supervisor",
                model_endpoint=settings.SUPERVISOR_URL,
                content=answer[:200] + "..." if len(answer) > 200 else answer,
                token_count=response.get("token_count")
            ))
            
            return answer
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return "I encountered an error generating the answer. Please try again."

    def _render_final_answer(self, answer_data: Dict[str, Any], evidence_bundle: EvidenceBundle) -> str:
        """Render structured final answer while rejecting fabricated citations."""
        allowed_chunk_ids = {passage.chunk_id for passage in evidence_bundle.passages}
        cited_items = []
        for item in answer_data.get("evidence_used", []):
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id in allowed_chunk_ids:
                cited_items.append(f"- {chunk_id}: {item.get('use', 'supporting evidence')}")

        sections = [str(answer_data.get("answer", "")).strip()]
        if cited_items:
            sections.append("Evidence used:\n" + "\n".join(cited_items))
        else:
            sections.append("Evidence used: No valid retrieved chunk citations were provided by the model.")
        uncertainty = answer_data.get("uncertainty")
        if uncertainty:
            sections.append(f"Uncertainty: {uncertainty}")
        risks = [str(item) for item in answer_data.get("contraindications_risks", []) if item]
        if risks:
            sections.append("Contraindications/risks:\n" + "\n".join(f"- {item}" for item in risks))
        next_steps = [str(item) for item in answer_data.get("next_steps", []) if item]
        if next_steps:
            sections.append("Next steps:\n" + "\n".join(f"- {item}" for item in next_steps))
        return "\n\n".join(part for part in sections if part)

    def _build_citations(self, evidence_bundle: EvidenceBundle) -> List[Dict[str, Any]]:
        """Build rich citations from selected evidence and ledger facets."""
        facets_by_chunk = {
            entry.chunk_id: entry.facets for entry in evidence_bundle.evidence_ledger
        }
        return [build_citation(passage, facets_by_chunk.get(passage.chunk_id, [])) for passage in evidence_bundle.passages]

    def _insufficient_evidence_answer(self, query: str, decision) -> str:
        """Return bounded CDS response when enterprise policy gates fail."""
        missing = ", ".join(decision.missing_facets) if decision.missing_facets else "required clinical facets"
        conflicts = " Unresolved high-severity contradictions were detected." if decision.unresolved_critical_conflicts else ""
        return ensure_cds_language(
            "The available evidence is insufficient to provide a grounded clinician decision-support answer for "
            f"the query: {redact_phi_text(query)}. Missing or uncertain evidence: {missing}.{conflicts} "
            "The next step is targeted evidence retrieval or clinician review of the relevant EMR/guideline source."
        )
