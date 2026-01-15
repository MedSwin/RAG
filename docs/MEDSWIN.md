# MedSwin Architecture Guide

## Overview

MedSwin operationalizes clinical QA as an **evidence-first, auditable pipeline** with Multi-Agent Conversation (MAC) orchestration. The system uses a supervisor-led architecture with specialist agents to provide evidence-based medical answers with full provenance.

## Architecture Components

### 1. Multi-Agent Conversation (MAC)

MedSwin uses a **supervisor/subagents** orchestration pattern:

#### Supervisor (Port 8000)
- Owns orchestration and evidence sufficiency policy checks
- Normalizes queries (canonical terms, abbreviations, retrieval hints)
- Performs safety critique
- Generates final answer with citations

#### Agent 1 - Evidence Retriever (Port 8001)
- Tool-using agent that searches CPG + EMR indices
- Iteratively requests `retrieve_more` if sufficiency not met
- Uses structured tool calls for retrieval

#### Agent 2 - EMR Summariser (Port 8002)
- Produces structured patient state:
  - Timeline
  - Problems
  - Medications
  - Allergies
  - Vitals
  - Labs
  - Contraindications flags

#### Agent 3 - Guideline Synthesiser (Port 8003)
- Extracts actionable recommendations
- Identifies contraindications
- Extracts guideline strength/grade if present

### 2. Retrieval Pipeline

#### Two-Stage Retrieval

**Stage 1: Candidate Retrieval**
- Dense vector search using dynamic index (HNSW/FAISS/Tree)
- Optional BM25 lexical union for rare terms/abbreviations
- Returns candidate pool (default K=80, expanded K'=120)

**Stage 2: Pointwise Reranking**
- Calls reranker service (port 8004) for `(query, passage)` scoring
- Produces calibrated probability `p_hat(q,d)`
- Calibration params stored in config/MongoDB

#### Fusion Scoring

Final selection score `S(q,d)` uses convex combination:
- `W_RERANK * p_hat(q,d)` (reranker probability)
- `W_DENSE * normalized_dense_similarity`
- `W_LEX * normalized_lexical_score` (if BM25 enabled)
- `W_RECENCY * recency_score`
- `W_SECTION * section_score` (recommendations > background)
- `W_SOURCE * source_prior` (CPG vs EMR weighting)

#### MMR Selection

Maximal Marginal Relevance (MMR) selection under:
- Token budget `B` (default 1800 tokens)
- Evidence sufficiency constraints for EMR & CPG coverage
- Diversity penalty to avoid redundant passages

### 3. Evidence Sufficiency Policy

Deterministic sufficiency gates:
- Targets: `T_CPG`, `T_EMR` (default: 2 each)
- Coverage ratios: `kappa_CPG >= 1`, `kappa_EMR >= 1`
- Mean confidence: `>= t'` (default: 0.60)

If not satisfied:
- Triggers `retrieve_more` action:
  - Increases candidate pool size (K → K')
  - Relaxes metadata filters (within policy)
  - Expands synonyms
- Loops up to `MAX_RETRIEVE_LOOPS` (default: 3)

If still insufficient after loops:
- Returns "insufficient evidence" response
- Asks clarifying questions (no confident diagnosis)

### 4. Enterprise Features

#### Audit & Provenance
Every chat request produces a durable trace in MongoDB:
- `session_id`, `user_id`, `org_id` (multi-tenant ready)
- Timestamps
- Agent messages (role, agent_id, model endpoint, token counts)
- Tool calls + tool outputs
- Selected evidence bundle (chunk_ids, doc_ids, source types)
- Sufficiency checks per loop iteration
- Final answer + citations mapping

#### Security
- API key / JWT auth (feature-flag enabled)
- Role-based access control (RBAC) for:
  - Ingestion
  - Patient EMR retrieval
  - Admin endpoints
- Data separation by `org_id`
- PHI-safe logging (redacts PHI fields unless debug mode)

#### Observability
- Structured JSON logging
- `/health`, `/metrics` endpoints
- RAG metrics: retrieval latency, rerank latency, loops triggered, sufficiency pass rate
- OpenTelemetry hooks (optional, behind env flag)

## Data Model

### MongoDB Collections

1. **documents**: Document metadata
   - `doc_id`, `source_type` (CPG|EMR|LIT), `title`, `version`, `effective_date`, `patient_id?`, `org_id`, `tags`

2. **chunks**: Text chunks with metadata
   - `chunk_id`, `doc_id`, `source_type`, `text`, `section`, `offset_start`, `offset_end`
   - Metadata: `patient_id?`, `guideline_version?`, `timestamp?`, `org_id`
   - Lexical fields for BM25 (if enabled)

3. **sessions**: User sessions
   - `session_id`, `user_id`, `org_id`, `created_at`, `last_active`

4. **traces**: Full audit traces
   - Complete request trace with messages, tool calls, evidence bundle, sufficiency checks

5. **embeddings**: Vector embeddings (optional separate collection)
   - `chunk_id`, `vector`, `dim`, `model_id`, `org_id`

6. **indices**: Index metadata
   - Stored index metadata (type, file paths, build time, params)

## API Endpoints

### POST /api/v1/medswin/chat

Process a chat query through the MedSwin pipeline.

**Request:**
```json
{
  "query": "What are the treatment guidelines for type 2 diabetes?",
  "user_id": "user123",
  "org_id": "org456",
  "session_id": "optional_session_id",
  "patient_id": "optional_patient_id",
  "constraints": {
    "guideline_only": true,
    "timeframe": "2024",
    "specialties": ["endocrinology"]
  }
}
```

**Response:**
```json
{
  "answer": "Based on the evidence...",
  "evidence_bundle": {
    "passages": [...],
    "total_tokens": 1200,
    "cpg_count": 3,
    "emr_count": 2,
    "lit_count": 0
  },
  "safety_notes": null,
  "trace_id": "trace_123",
  "degraded_mode": {},
  "uncertainty_level": "medium",
  "citations": [
    {
      "chunk_id": "chunk_001",
      "doc_id": "doc_001",
      "source_type": "CPG",
      "section": "Recommendations"
    }
  ]
}
```

### GET /api/v1/medswin/sessions/{session_id}

Get session summary and last N turns (redacted).

### GET /api/v1/medswin/traces/{trace_id}

Get structured trace (admin/authorized only, optionally redacted).

### POST /api/v1/medswin/ingest

Ingest CPG or EMR documents with proper metadata.

## Configuration

All configuration is environment-driven. See `env.example` for complete list.

### Key Configuration Groups

1. **Model Endpoints**: All LLM, reranker, and embedding endpoints
2. **Retrieval Knobs**: Candidate K, token budget, BM25 enable
3. **Sufficiency Policy**: Targets, thresholds, max loops
4. **Fusion Weights**: Must sum to 1.0
5. **Enterprise Flags**: Auth, RBAC, PHI redaction

## Response Formatting Rules

The final user-facing answer **must** include:
- "Evidence used" section referencing chunk_ids/doc_ids
- Explicit uncertainty language
- Contraindications/risks when present in CPG

The system **must not** fabricate citations: only cite retrieved chunks.

If evidence is insufficient after loops:
- Do **not** guess; ask targeted clarifying questions

## Integration with Dynamic Indexing

MedSwin leverages the existing dynamic indexing infrastructure:
- Uses `IndexStrategyManager` for index selection
- Supports HNSW, FAISS IVF, and Tree indices
- BM25 union works alongside vector search
- Reranker integrates with all index types

See [INDEXING.md](INDEXING.md) for details on dynamic indexing.

## Troubleshooting

### Model Endpoints Not Available

If model endpoints are down:
- Reranker down → Falls back to dense+lex fusion only (logs degraded mode)
- Agent endpoint down → Supervisor continues with remaining agents (logs partial mode)
- If critical agent down → Returns "insufficient evidence" response

### Sufficiency Loops Not Triggering

Check:
- `SUFF_T_CPG` and `SUFF_T_EMR` targets
- `SUFF_T_MEAN_CONF` threshold
- `MAX_RETRIEVE_LOOPS` limit
- Evidence coverage in database

### Performance Issues

- Increase `CANDIDATE_K` for better recall
- Enable BM25 for rare terms
- Adjust fusion weights based on domain
- Monitor token budget usage

