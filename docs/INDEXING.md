# Indexing and embeddings

How MedSwin (and the naive-RAG control) turn ingested documents into retrievable vectors. This page describes the **live chat path**, not the older “pick HNSW or FAISS or Tree per query” sketch.

Admin first-run: [ADMIN.md](ADMIN.md).  
Local rebuild from the operator: `./scripts/start-local.sh index`.  
API: [ENDPOINTS.md](ENDPOINTS.md).  
Fairness rules: [NAIVE_RAG.md](NAIVE_RAG.md).

---

## 1. What chat actually queries

`DenseRetriever` (`app/retrieval/dense.py`) embeds the query, then calls `HybridIndex` (`app/indexes/hybrid.py`):

```text
query vector
    → HNSW  (HNSW_INDEX_PATH + HNSW_MAPPING_PATH)
    → FAISS IVF  (FAISS_INDEX_PATH + FAISS_MAPPING_PATH)   if the file exists
    → merge by dense similarity
    → hydrate chunk rows from Mongo (org / patient / source / grade filters)
```

Both indexes, when present, are queried. Candidates are the **union**, scored by the better dense similarity. There is no per-query “strategy manager” on the MedSwin chat path.

Full MedSwin then unions those dense hits with BM25 (`ENABLE_BM25`, `app/retrieval/lexical.py`) and reranks. Naive-RAG stops after dense top-K (`NAIVE_TOP_K`).

Default local files (see `env.example`):

```text
./data/hnsw_index.bin
./data/hnsw_mapping.json
./data/faiss_index.bin
./data/faiss_mapping.json
```

The mapping files store `integer_label → chunk_id`. The eval harness reads the HNSW provenance manifest and refuses to score a run whose index belongs to another org or embedding space.

---

## 2. Builders that still exist

`app/core/indexing/` still contains:

| Builder | File | Role |
| --- | --- | --- |
| HNSW | `HNSWIndexBuilder` | Default ANN for chat |
| FAISS IVF | `FAISSIndexBuilder` | Second ANN; merged at query time |
| BallTree | `TreeIndexBuilder` | Legacy / ops; **not** queried by `HybridIndex` |

`INDEX_STRATEGY_MODE=dynamic` and `DEFAULT_INDEX_TYPE=hnsw` affect **index build** helpers and legacy `/retrieval` routes. They do not switch MedSwin chat away from HNSW ∪ IVF.

---

## 3. Ingest attaches embeddings

`POST /api/v1/medswin/ingest?source_type=…&org_id=…`

1. Validates `source_type` ∈ {CPG, EMR, LIT, SAFETY}
2. Section-aware chunking if `chunks` is omitted (`app/medswin/chunking.py`, ~350–450 tokens)
3. Writes documents + chunks scoped by `org_id` (EMR also stores `patient_id`)
4. **Always attempts** `_attach_active_embeddings`:
   - `CLOUD_MODE`: embed via the cloud client; failure is fatal
   - local mode: embed via `EmbeddingClient`, which falls back to the loaded `ModelManager` when `EMBEDDING_URL` is down; failure is a warning and chunks are still stored

Chunks without vectors are invisible to ANN and to naive Mongo cosine (that scan requires `embedding: {$exists, $type: array}`). `GET /api/v1/naive/ready` reports `chunk_count` vs `embedded_count`. If those diverge, do not publish a compare — refresh.

---

## 4. Refresh and rebuild

```bash
# attach / replace vectors for the active embedding space
curl -s http://127.0.0.1:8100/api/v1/storage/embeddings/refresh \
  -H 'Content-Type: application/json' \
  -d '{"org_id":"demo-org"}'

# rebuild ANN files from those vectors
curl -s http://127.0.0.1:8100/api/v1/storage/index/build \
  -H 'Content-Type: application/json' \
  -d '{"force_rebuild":true,"org_id":"demo-org"}'
```

Or: `./scripts/start-local.sh index --org-id demo-org`.

`refresh` now embeds in **local** mode as well as cloud mode. Cloud refresh is batched (`CLOUD_EMBED_BATCH_SIZE`, `CLOUD_EMBED_BATCH_DELAY_S`) to respect quota. Local refresh does not inject that 60s delay.

After a successful build, `GET /api/v1/storage/stats?org_id=…` should show:

- `index_exists=true`
- `index_manifest.org_id` matching the caller
- `index_manifest.embedding_model` / `embedding_dim` matching `active_embedding_*`
- `total_vectors` matching `active_embeddings`
- LIT source counts > 0 for a literature benchmark

The eval runner (`eval/app/runner.py`) fails the run if those provenance checks fail.

`POST /api/v1/storage/benchmark/reset` clears a benchmark org (default `bench-org`) and optionally deletes index files before a fresh TREC ingest.

---

## 5. Filters applied at hydration

`app/retrieval/filters.py` builds the Mongo filter used by both pipelines:

- always `org_id`
- `source_type` when `guideline_only` / `source_policy` is CPG/EMR/LIT/SAFETY
- `patient_id` on EMR
- `min_evidence_grade` when the client sends it (eval default 0.3)

Naive-RAG applies the same filter. That is intentional: the control must not see another org’s chunks.

Dimension mismatches are skipped. Naive then reports `retrieval_backend=dim_mismatch` and refuses to stuff zero-score junk into the prompt.

---

## 6. Naive fallback (not for publication)

If ANN returns nothing and `NAIVE_ENABLE_MONGO_FALLBACK=true`, naive scans up to `NAIVE_MONGO_SCAN_LIMIT` embedded chunks and ranks by cosine. `retrieval_backend` becomes `mongo_cosine`.

This exists so `./scripts/start-local.sh ask --mode naive` still works before you build an index. A number you would write in a paper requires `retrieval_backend=ann`.

Empty retrieval with chunks but `embedded_count==0` is an **infrastructure** failure: naive does not call the LLM and sets `degraded_mode.no_embeddings`. True empty retrieval (no chunks) still parametric-generates — that is the ungated control.

---

## 7. BM25

Enabled with `ENABLE_BM25=true`. Tokenized text lives on the chunk. Lexical retrieval is MedSwin-only. Naive must not call it.

---

## 8. Practical sequence after a clean checkout

```bash
./scripts/start-local.sh serve          # or console
# ingest LIT / CPG / EMR for demo-org or bench-org
./scripts/start-local.sh index --org-id demo-org
curl -s http://127.0.0.1:8100/api/v1/naive/ready
./scripts/start-local.sh ask --mode both --question "Can metformin continue?"
```

Confirm `retrieval_backend=ann` on the naive side before you treat the diff as a design result.
