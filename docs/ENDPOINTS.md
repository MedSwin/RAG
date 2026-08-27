# MedSwin HTTP API

Runtime base URL: `http://127.0.0.1:8100`. All versioned routes live under `/api/v1`.

The production data contract is tenant scoped. Any route that reads or mutates corpus data requires an `org_id` directly or inside its request body. Cloud mode uses Azure AI Foundry adapters; local mode uses the configured/local Hugging Face model services.

## Process and health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Redirect to clinician UI when built, otherwise dashboard |
| `GET` | `/health` | Mongo-backed liveness plus active model/embedding identities |
| `GET` | `/app/` | Clinician CDS UI |
| `GET` | `/docs` | OpenAPI |

Cloud startup validates the provider configuration before serving traffic. `paper-eval` warmup performs the stronger live Foundry/model/corpus probes before a publication run.

## MedSwin (`/api/v1/medswin`)

### `POST /medswin/chat`

```json
{
  "query": "Can this patient continue metformin?",
  "user_id": "clinician-1",
  "org_id": "demo-org",
  "session_id": null,
  "patient_id": "patient-42",
  "constraints": {
    "clinical_scope": "clinician_cds",
    "source_policy": "ANY",
    "min_evidence_grade": 0.3
  }
}
```

Full path: query normalization → dense ANN ∪ BM25 → Cohere/local rerank → MAC specialists → sufficiency/policy gate → synthesis or bounded abstention. Insufficient evidence is HTTP 200 with `policy_decision.passed=false`.

### `POST /medswin/ingest?source_type=LIT&org_id=demo-org`

Body is a JSON array of documents. If `chunks` is omitted, section-aware chunking runs. In cloud mode every staged chunk is embedded with **document** intent before Document/Chunk persistence begins; a cloud embedding failure therefore does not leave a metadata-only partial document. Local mode may store text when the local embedding runtime is unavailable and can be repaired with the storage refresh route.

### Sessions and traces

- `GET /medswin/sessions/{session_id}?org_id=...`
- `GET /medswin/traces/{trace_id}?org_id=...&include_details=false`

Both use tenant-scoped repositories. Trace output is PHI-redacted by default.

## Naive RAG (`/api/v1/naive`)

- `GET /naive/ready`
- `POST /naive/chat`
- `POST /naive/compare`

The control is intentionally: query embedding → dense ANN top-K → shared-budget context → one generation. It has no BM25, reranker, MAC specialists, or sufficiency gate. Publication evaluation rejects ANN fallback/degraded control runs.

## Storage and indexing (`/api/v1/storage`)

The online application keeps one shared HNSW artifact containing active vectors from all ordinary tenants; ANN labels are resolved through tenant-filtered Mongo queries. `paper-eval` uses isolated artifact paths and its separate streaming full-corpus builder.

### `POST /storage/chunks`

Accepts either current production chunks or the legacy preprocessing response shape. Storage normalizes the record, generates any missing/stale active embedding using **document** intent, writes it idempotently under `(org_id, chunk_id)`, then rebuilds the ordinary global ANN in a background task.

```json
{
  "org_id": "demo-org",
  "source_type": "LIT",
  "batch_size": 64,
  "chunks": [
    {
      "content": "legacy preprocessing output is accepted",
      "metadata": {
        "chunk_id": "example-1",
        "parent_id": "doc-1"
      }
    }
  ]
}
```

Current-schema chunks may instead provide top-level `chunk_id`, `doc_id`, `text`, `source_type`, metadata, and an already-valid active embedding.

### `POST /storage/embeddings/refresh`

Refreshes stale vectors for one tenant without publishing a tenant-only ANN. A global rebuild is queued only after the tenant refresh succeeds.

```json
{"org_id":"demo-org","batch_size":64}
```

### `POST /storage/index/build`

Rebuilds the one ordinary global HNSW artifact from all active vectors. The builder refuses oversized in-memory corpora (`STORAGE_IN_MEMORY_INDEX_MAX_VECTORS`, default 250000); the 1.25M TREC corpus must use `paper-eval`'s streaming builder.

```json
{"force_rebuild":true}
```

### Other storage routes

| Method | Path | Scope |
| --- | --- | --- |
| `GET` | `/storage/stats?org_id=...` | tenant stats plus index provenance |
| `GET` | `/storage/chunks?org_id=...` | tenant list, bounded pagination |
| `GET` | `/storage/chunks/{chunk_id}?org_id=...` | tenant lookup |
| `DELETE` | `/storage/chunks?org_id=...` | delete only that tenant's chunks |
| `POST` | `/storage/validate?org_id=...` | chunk/index validation |
| `POST` | `/storage/benchmark/reset` | benchmark-org reset + optional isolated artifacts |

## Embedding (`/api/v1/embedding`)

These utility endpoints now follow the active runtime instead of assuming local Hugging Face weights exist.

- `POST /embedding/embed`
- `POST /embedding/embed/batch`
- `GET /embedding/info`

Example:

```json
{"text":"clinical search text","input_type":"query","normalize":true}
```

`input_type` is `query` or `document`. Online cloud retrieval defaults to query intent; corpus-writing paths pass document intent explicitly.

## Retrieval (`/api/v1/retrieval`)

- `POST /retrieval/search`
- `GET /retrieval/search`
- `GET /retrieval/index/info`

The search route uses the same `EmbeddingClient`, `DenseRetriever`, tenant/patient/source filters, HNSW loader (JSON or SQLite mapping), and optional `RerankerClient` used by MedSwin. It does **not** execute MAC/gating and therefore is not the full CDS pipeline.

```json
{
  "query": "renal metformin guidance",
  "org_id": "demo-org",
  "patient_id": null,
  "top_k": 10,
  "use_reranking": true,
  "source_type": "LIT",
  "constraints": {}
}
```

## Preprocessing (`/api/v1/preprocessing`)

- `POST /preprocessing/chunk`
- `POST /preprocessing/upload-and-chunk`
- `GET /preprocessing/preprocessing/info`
- `POST /preprocessing/validate-chunks`

Cloud mode uses a local `tiktoken` tokenizer for chunk accounting, so these endpoints do not require local embedding weights. CSV/JSON upload is supported by this dialogue preprocessor. Its legacy `{content, metadata}` chunks can be sent directly to `/storage/chunks`, which performs production-schema normalization and embedding.

## Dashboard (`/api/v1/dashboard`)

Operations/dashboard routes remain separate from clinician CDS. Dataset preloading can be disabled with `DISABLE_DATASET_PRELOAD=true`; paper-eval disables it to avoid benchmark noise.

## Evaluation

There is no `:8200` eval service. Official TREC CDS 2016 evaluation is:

```bash
./scripts/start-local.sh paper-eval
```

T1 is a LIT-only retriever exporter (not `/chat`). T3/T4 use the product chat path on `FULL_EVAL_API_PORT` (default 8110). See [PAPER_EVAL.md](PAPER_EVAL.md).

## Authentication

`ENABLE_AUTH=false` remains the default development configuration. When enabled, the current middleware enforces Bearer-token presence but is still an identity-provider integration scaffold; do not represent it as completed enterprise JWT/RBAC authorization.
