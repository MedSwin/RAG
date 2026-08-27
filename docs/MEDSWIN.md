# MedSwin runtime architecture

This page describes the **current** executable system in `app/medswin/`. It replaces the older supervisor-on-port-8000 / three-tool-agent sketch.

Paper source (not tracked in git): `docs/MedSwin.tex`  
HTTP contract: [ENDPOINTS.md](ENDPOINTS.md)  
Naive control: [NAIVE_RAG.md](NAIVE_RAG.md)  
Admin runbook: [ADMIN.md](ADMIN.md)  
Local operator: [OPERATOR.md](OPERATOR.md)  
Publication eval: [PAPER_EVAL.md](PAPER_EVAL.md)

---

## 1. Decision the system is allowed to make

Generic RAG asks which passages to stuff into a prompt.

MedSwin asks whether the available evidence is **enough** to support a clinician-facing answer. Top-K similarity does not authorize generation. The sufficiency gate does.

Outcomes:

| Action | Meaning |
| --- | --- |
| `accept` | Critical facets pass LCB, entropy, and contradiction constraints → synthesis |
| `retrieve_more` | Gate failed; loop budget remains; hints widen or focus retrieval |
| `insufficient_evidence` | Gate failed and further retrieval is exhausted or low-utility → bounded CDS abstention |

The answer is always clinician decision support, never a final diagnosis or order.

---

## 2. Request path

```text
Clinician UI / operator / paper-eval T3 packs
        │
        ▼
POST /api/v1/medswin/chat
        │
        ▼
MedSwinOrchestrator.chat
        │
        ├─ SessionRepository  (create or touch session)
        ├─ QueryNormalizer    (canonical terms, facets, optional EMR synopsis)
        │
        └─ retrieve-more loop (MAX_RETRIEVE_LOOPS, default 3)
              HybridRetriever.retrieve     HNSW ∪ IVF  ∪  BM25
              HybridRetriever.rerank       pointwise + Platt/T
              MAC claim agents             EMR / Guideline / Safety / Quality / Critic
              claim ledger + contradictions
              fusion + utility selection + packing
              EvidenceGate
                 ACCEPT        → SynthesisAgent (citation-hardened)
                 RETRIEVE_MORE → hints / expand_query / loop
                 INSUFFICIENT  → abstain renderer
        │
        ▼
AuditTrace persisted  →  ChatResponse  →  GET /medswin/traces/{id}
```

The naive-RAG control is a **different orchestrator** (`NaiveRAGOrchestrator`) on `POST /api/v1/naive/chat`. It reuses embeddings, ANN, Mongo, filters, and the LLM. It does not call this loop. See [NAIVE_RAG.md](NAIVE_RAG.md).

---

## 3. Multi-agent coordination (MAC)

Orchestration is **centralized**. Agents do not peer-negotiate. They emit structured claims:

```text
(facet, claim, polarity, chunk_id, evidence_grade, calibrated_relevance, provenance)
```

| Agent | Module | Job | Typical missing-facet route |
| --- | --- | --- | --- |
| Normalize | `app/medswin/normalize.py` | Query → `QuerySpec` + facet set \(F_q\) | always first |
| EMR | `app/agents/emr.py` | Meds, labs, comorbidities, allergies | `patient_applicability` |
| Guideline | `app/agents/guideline.py` | CPG recs, eligibility, strength / version | `guideline_concordance` |
| Safety | `app/agents/safety.py` | Contraindications, AEs, DDI, dose limits | `safety_contraindications` |
| Quality | `app/agents/quality.py` | Grade, recency, population fit, provenance | `evidence_quality` |
| Critic | `app/agents/critic.py` | High-severity conflicts vs outdated / mismatch | contradiction review |
| Synthesis | `app/agents/synthesis.py` | Final CDS answer **only** from the accepted bundle | after `accept` |

Reliability priors \(r_i \sim \mathrm{Beta}(a_i, b_i)\) become LCB weights \(\omega_i\) from [`data/calibration/agents.json`](../data/calibration/agents.json).

`SUPERVISOR_URL` / `AGENT1_URL`–`AGENT3_URL` still exist in config. In `CLOUD_MODE=true` they collapse to one Azure chat deployment. Locally, missing agent endpoints set `degraded_mode.agent_*` and the loop continues fail-open.

---

## 4. Two-stage retrieval and scoring

```text
query (+ optional patient synopsis)
    → Dense: HybridIndex  (HNSW ∪ FAISS/IVF over the same org vectors)
    → Lexical: BM25 if ENABLE_BM25
    → union TopK / TopK′
    → pointwise rerank
    → p̂ = σ((ℓ − b) / T)     CalibrationStore, data/calibration/rerank.json
    → log-odds fusion + EBM − noise
    → greedy ΔU / τ under token budget B
    → protected packing (safety / contradiction / required-source)
```

Primary modules: `app/retrieval/hybrid.py`, `app/retrieval/dense.py`, `app/retrieval/lexical.py`, `app/indexes/hybrid.py`, `app/scoring/{fusion,utility,coverage,calibrate,hierarchy}.py`, `app/medswin/packing.py`.

Index files and rebuild: [INDEXING.md](INDEXING.md).

### Gate math (implemented)

Coverage is noisy-OR over selected passages, with bootstrap LCB (`SUFF_BOOTSTRAP_N=64`) and an entropy cap.

Generation requires, for every **critical** facet:

- lower confidence bound \(\ge\) facet \(\theta\) (`SUFF_CRITICAL_FACET_THRESHOLD`, default 0.78)
- entropy \(\le\) `SUFF_MAX_ENTROPY` (default 0.88)
- unresolved high-severity contradictions \(\le\) `SUFF_MAX_CONTRADICTIONS` (default 0)

`retrieve_more` also requires marginal utility \(\ge\) `SUFF_MIN_MARGINAL_UTILITY`.

Legacy count gates `SUFF_T_CPG` / `SUFF_T_EMR` remain in `env.example` for older code paths. The live gate in `app/medswin/gate.py` is facet-LCB, not “at least two CPG chunks.”

If `rerank.json` is missing or identity `{b:0,T:1}`, chat sets `degraded_mode.calibration=true` and continues.

---

## 5. Audit artefacts (per query)

| # | Artefact | Where |
| --- | --- | --- |
| 1 | Retrieval trace | dense / lex counts, hints, candidate ids |
| 2 | Rerank trace | calibrated scores, selected / rejected chunk ids |
| 3 | Facet-coverage matrix | status, LCB, \(\pi\), entropy |
| 4 | Contradiction ledger | high-severity pairs and resolution |
| 5 | Sufficiency decision | action, missing facets, routed agent |
| 6 | Answer provenance | statement → accepted `chunk_id`; rejected fabricated ids |

Returned on `ChatResponse` and on `GET /api/v1/medswin/traces/{trace_id}?org_id=…&include_details=true`. Default trace summaries redact PHI (`TRACE_REDACT_BY_DEFAULT`).

`ChatResponse.pipeline` is `"medswin"` for this orchestrator and `"naive_rag"` for the control.

---

## 6. Data scoped by tenant

Mongo database name: **`medswin`** (`MONGODB_DB`).

| Collection (logical) | Scope |
| --- | --- |
| documents | `org_id`, `source_type` ∈ {CPG, EMR, LIT, SAFETY} |
| chunks | `org_id`, optional `patient_id` on EMR, `embedding[]`, `embedding_space` |
| sessions | `org_id`, `user_id` |
| traces | `org_id`, `session_id`, full audit |

EMR hard-filters on `patient_id` for EMR-only / patient-scope queries. Mixed CDS cannot treat another patient’s note as literature.

---

## 7. Degradation and fail-open

The runtime prefers an explicit `degraded_mode` flag over a silent 500.

| Condition | Behaviour |
| --- | --- |
| Reranker down | Fusion without calibrated \(\hat p\); `degraded_mode.rerank` |
| Calibration identity | `degraded_mode.calibration` |
| One specialist LLM down | Remaining agents run; `degraded_mode.agent_*` |
| Embedder down at ingest | Local mode warns and stores chunks; naive later reports `no_embeddings` |
| Evidence insufficient | HTTP 200, `policy_decision.passed=false`, bounded abstention text |

Connection-refused to a model URL is **not** treated as a 429. The rate limiter fails those after two short retries so a down embedder cannot stall a prompt for hours.

---

## 8. Security and CDS boundary

| Flag / rule | Reality today |
| --- | --- |
| `ENABLE_AUTH` | Bearer scaffold only; not a full IdP JWT stack |
| `ENABLE_RBAC` | Flag present; not a complete role matrix |
| `TRACE_REDACT_BY_DEFAULT` | Emails, phones, MRNs, DOBs stripped from trace summaries |
| Org isolation | Enforced on sessions, traces, chunks, retrieval filters |
| Citation hardening | Synthesis may only cite `chunk_id`s in the accepted bundle |
| CDS language | `ensure_cds_language` appends the clinician-support boundary |

---

## 9. Configuration groups

See [`env.example`](../env.example) and `app/core/config.py`.

1. Process: `APP_HOST`, `APP_PORT` (**8100**)
2. Mongo: `MONGODB_URL`, `MONGODB_DB=medswin`
3. Models: `CLOUD_MODE` or `SUPERVISOR_URL` / `EMBEDDING_URL` / `RERANKER_URL`
4. Retrieval: `CANDIDATE_K`, `CANDIDATE_K_PRIME`, `ENABLE_BM25`, `TOKEN_BUDGET_B`
5. Gate: `SUFF_FACET_THRESHOLD`, `SUFF_CRITICAL_FACET_THRESHOLD`, `SUFF_MAX_ENTROPY`, `SUFF_MAX_CONTRADICTIONS`, `SUFF_MIN_MARGINAL_UTILITY`
6. Naive control: `NAIVE_TOP_K`, `NAIVE_MAX_CONTEXT_CHARS`, `NAIVE_ENABLE_MONGO_FALLBACK`
7. Fusion weights: `W_RERANK`, `W_DENSE`, `W_LEX`, `W_EBM`, `W_NOISE`, …
8. Enterprise: `ENABLE_AUTH`, `TRACE_REDACT_BY_DEFAULT`

Invalid enterprise policy thresholds fail startup.

---

## 10. Troubleshooting (runtime)

| Symptom | Check |
| --- | --- |
| Sufficiency never accepts | Facet LCBs in the trace; thresholds; empty LIT / CPG for that org |
| Retrieve-more loops burn the budget | `MAX_RETRIEVE_LOOPS`, `SUFF_MIN_MARGINAL_UTILITY`, hint `focus_source` |
| Answers ignore the patient | EMR ingest with matching `patient_id` and `org_id` |
| Citations look fabricated | Should be impossible after synthesis hardening; inspect `answer_provenance.rejected_chunk_ids` |
| Slow compares | Full MAC + up to 3 retrieve loops; operator `--timeout` default is 300s |
