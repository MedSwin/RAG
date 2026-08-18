# MedSwin HTTP API

Base URL for the runtime: `http://127.0.0.1:8100`

All versioned routes sit under `/api/v1`. Every MedSwin / naive / storage call that reads tenant data requires `org_id`.

Interactive explorer: [http://127.0.0.1:8100/docs](http://127.0.0.1:8100/docs)  
Local operator: [OPERATOR.md](OPERATOR.md)  
Architecture: [MEDSWIN.md](MEDSWIN.md)

---

## Process and pages

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Redirect to `/app/` when the clinician UI is present, else `/api/v1/dashboard/` |
| `GET` | `/health` | Liveness. Local: embedding / reranker loaded. Cloud: active embedding space + refresh status |
| `GET` | `/app/` | Clinician CDS (static `web/public` or built `web/dist`) |
| `GET` | `/docs` | OpenAPI |

`/health` is **not** namespaced under `/api/v1`.

---

## MedSwin (`/api/v1/medswin`)

### `POST /api/v1/medswin/chat`

Full MAC + gate pipeline.

```json
{
  "query": "Can this patient continue metformin after the latest renal-function result?",
  "user_id": "clinician-1",
  "org_id": "demo-org",
  "session_id": "optional",
  "patient_id": "patient-42",
  "constraints": {
    "clinical_scope": "clinician_cds",
    "guideline_only": false,
    "required_facets": [],
    "source_policy": "ANY",
    "min_evidence_grade": 0.3
  }
}
```

`source_policy` ∈ `ANY` | `CPG_ONLY` | `EMR_ONLY` | `LIT_ONLY`.

Response is a `ChatResponse`:

| Field | Meaning |
| --- | --- |
| `answer` | Clinician CDS text; never a final diagnosis |
| `pipeline` | `"medswin"` |
| `policy_decision` | Gate: `passed`, `action`, `reason` |
| `evidence_bundle` | Selected passages, source counts, ledger, facet coverage |
| `facet_coverage` | Per-facet LCB / entropy / status |
| `contradictions` | High-severity pairs |
| `evidence_ledger` | Claim-level provenance |
| `citations` | `chunk_id`, `doc_id`, `source_type`, section, version |
| `trace_id` | Fetch the full audit with the traces route |
| `degraded_mode` | Service-failure flags (rerank, calibration, agents, …) |
| `uncertainty_level` | Model-facing uncertainty hint |
| `retrieval_backend` | Present on some paths; naive sets this explicitly |
| `timing_ms` | Optional stage timings |

Insufficient evidence is still HTTP **200** with `policy_decision.passed=false` and a bounded abstention answer.

### `GET /api/v1/medswin/sessions/{session_id}`

Query: `org_id` (required).

Returns `session_id`, `user_id`, `org_id`, `created_at`, `last_active`, `metadata`.

### `GET /api/v1/medswin/traces/{trace_id}`

Query:

- `org_id` (required)
- `include_details` (default false). Policy artefacts are included only when this is true **and** `TRACE_INCLUDE_POLICY_DETAILS` is true on the server.

Always returns counts: `messages_count`, `tool_calls_count`, `sufficiency_checks_count`, `evidence_passages_count`, plus a PHI-redacted query. Naive traces are stored in the same collection and are readable here.

### `POST /api/v1/medswin/ingest`

Query:

- `source_type`: `CPG` | `EMR` | `LIT` | `SAFETY`
- `org_id`: required

Body: JSON array of documents.

```json
[
  {
    "doc_id": "guideline-1",
    "title": "Guideline title",
    "version": "2026.1",
    "effective_date": "2026-01-01T00:00:00",
    "patient_id": "optional-for-emr",
    "source_reliability": 0.95,
    "evidence_grade": {"label": "guideline", "score": 0.95, "source_reliability": 0.95},
    "tags": ["diabetes"],
    "metadata": {},
    "text": "Recommendations\n\n...",
    "chunks": [
      {
        "chunk_id": "optional",
        "text": "Chunk text",
        "section": "Recommendations",
        "offset_start": 0,
        "offset_end": 120,
        "metadata": {}
      }
    ]
  }
]
```

If `chunks` is omitted, the server runs section-aware chunking. Ingest **always tries to attach** the active embedding space. Cloud mode fails the request if embed fails. Local mode warns and stores text so you can `POST /storage/embeddings/refresh` later. See [INDEXING.md](INDEXING.md).

---

## Naive-RAG control (`/api/v1/naive`)

Fairness contract: [NAIVE_RAG.md](NAIVE_RAG.md).

### `GET /api/v1/naive/ready`

Lightweight preflight. No `org_id` filter (global counts).

```json
{
  "pipeline": "naive_rag",
  "cloud_mode": false,
  "embedding_url": "...",
  "llm_url": "...",
  "index_exists": true,
  "mongo": true,
  "chunk_count": 12,
  "embedded_count": 12,
  "ready": true
}
```

`ready` is true when Mongo pings. Eval still fails the run if `chunk_count > 0` and `embedded_count == 0`.

### `POST /api/v1/naive/chat`

Same body as `/medswin/chat`, plus optional `top_k` (default `NAIVE_TOP_K`).

Pipeline: embed → dense ANN top-K → one generate. No BM25, rerank, MAC, or gate. `policy_decision.passed` is always true.

Extra response fields:

- `pipeline`: `"naive_rag"`
- `retrieval_backend`: `ann` | `mongo_cosine` | `empty` | `dim_mismatch` | `error`
- `timing_ms`: `{embed, retrieve, generate, total}`
- `uncertainty_level`: `"ungated"` (or high when degraded / empty)
- `degraded_mode`: `error`, `no_embeddings`, `empty_index`, `trace_persist`

Infrastructure gaps (`no_embeddings`, `dim_mismatch`) do **not** call the LLM.

### `POST /api/v1/naive/compare`

Same body. Runs naive then MedSwin and returns:

```json
{
  "query": "...",
  "naive": { "pipeline": "naive_rag" },
  "medswin": { "pipeline": "medswin" },
  "diff": {
    "jaccard": 0.0,
    "overlap_chunk_ids": [],
    "naive_only_chunk_ids": [],
    "medswin_only_chunk_ids": [],
    "medswin_abstained": true,
    "naive_backend": "ann",
    "timing_ms": { "naive": 1200.0, "medswin": 8400.0 }
  }
}
```

---

## Storage (`/api/v1/storage`)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/storage/stats` | Chunk / embedding / index provenance. Query `org_id` |
| `POST` | `/storage/index/build` | Rebuild HNSW (and FAISS when configured) from embedded chunks |
| `POST` | `/storage/embeddings/refresh` | Embed stale / missing vectors in the **active** space; local and cloud |
| `POST` | `/storage/benchmark/reset` | Clear `bench-org` (or given `org_id`) and optionally delete index files |
| `POST` | `/storage/chunks` | Legacy bulk chunk insert (triggers a background index rebuild) |
| `GET` | `/storage/chunks` | List chunks |
| `GET` | `/storage/chunks/{chunk_id}` | Fetch one chunk |
| `DELETE` | `/storage/chunks` | Clear chunks (dangerous) |
| `POST` | `/storage/validate` | Index / corpus validation snapshot |

Build body:

```json
{"force_rebuild": true, "org_id": "demo-org"}
```

Refresh body:

```json
{"org_id": "demo-org", "batch_size": 64}
```

Stats fields the eval harness depends on: `source_counts`, `index_exists`, `index_manifest`, `index_provenance_valid`, `active_embedding_model`, `active_embedding_dim`, `active_embeddings`, `active_doc_ids`.

---

## Ops dashboard (`/api/v1/dashboard`)

HTML UI at `/api/v1/dashboard/`. JSON helpers:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/dashboard/stats` | Hugging Face dataset totals (cached) |
| `GET` | `/dashboard/status` | Service status |
| `GET` | `/dashboard/models` | Local model presence |
| `POST` | `/dashboard/download-models` | Download embedding / reranker weights |
| `POST` | `/dashboard/ingest/{dataset_name}` | Legacy HF ingest job |
| `GET` | `/dashboard/dataset/{dataset_name}` | Dataset card |

This is an operations console, not the clinician CDS.

---

## Legacy RAG (`/api/v1/{preprocessing,embedding,retrieval}`)

Still mounted for compatibility. They do **not** run the sufficiency gate.

- `POST /preprocessing/chunk`
- `POST /preprocessing/upload-and-chunk`
- `GET /preprocessing/preprocessing/info` (legacy path on that router)
- `POST /preprocessing/validate-chunks`
- `POST /embedding/embed`
- `POST /embedding/embed/batch`
- `GET /embedding/info`
- `POST /retrieval/search`
- `GET /retrieval/search`
- `GET /retrieval/index/info`

Use `/medswin/chat` or `/naive/chat` for anything you would report.

---

## Eval harness (`http://127.0.0.1:8200`)

Separate FastAPI app: `eval.app.main:app`. Start with `./scripts/start-local.sh eval --open`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Benchmark UI (`eval/static/index.html`) |
| `GET` | `/health` | `{status, service: medswin-system-benchmark}` |
| `GET` | `/api/cases` | Preview a cases JSONL |
| `POST` | `/api/run` | Run an audit (`RunRequest`) |
| `GET` | `/api/runs` | List stored audits (skips `*.comparison.json`) |
| `GET` | `/api/runs/{run_id}` | Full `RunAudit` |
| `GET` | `/api/runs/{run_id}/download` | Download JSON |

`POST /api/run` body (defaults shown):

```json
{
  "cases_path": "eval/data/sample/cases.jsonl",
  "max_cases": 2,
  "max_concurrency": 1,
  "reranker_budget": 1,
  "ingest_case_context": true,
  "fetch_trace_summary": true,
  "include_patient_context_in_query": false,
  "source_policy": "ANY",
  "guideline_only": false,
  "min_evidence_grade": 0.3,
  "clinical_scope": "clinician_cds",
  "pipeline": "medswin",
  "top_k": 5
}
```

`pipeline` ∈ `medswin` | `naive_rag` | `both`.

`both` returns the MedSwin `RunAudit`. Naive aggregates and deltas are under `diagnostics.pipeline_comparison`. Sidecar: `{run_id}.comparison.json`.

Audits default to `RUN_STORE_DIR=/tmp/medswin-audits`. Full metric definitions: [`eval/README.md`](../eval/README.md).

---

## Auth

`ENABLE_AUTH=false` by default. When true, `AuthMiddleware` expects a Bearer token. This is a scaffold, not a production IdP integration.

---

## Error conventions

| Situation | HTTP | Body |
| --- | --- | --- |
| Insufficient evidence (MedSwin) | 200 | Abstention answer, `policy_decision.passed=false` |
| Naive infrastructure gap | 200 | Explanation in `answer`, `degraded_mode.no_embeddings` / `empty_index` |
| Unhandled orchestrator exception | 500 | `detail` string |
| Eval preflight (index / qrel / naive ready) | 500 from `/api/run` | Human-readable `detail` |

The operator CLI treats naive `degraded_mode.error` / `no_embeddings` as a failed one-shot ask (exit 1).
