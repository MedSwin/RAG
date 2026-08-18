# Naive-RAG baseline vs full MedSwin

This document is the operator manual for comparing the production MedSwin pipeline against a deliberately minimal **naive-RAG** control. It is written so a person who has never seen the repo can reproduce the setup, run both pipelines on the same question or the same case file, and interpret the difference.

Paper / architecture source: [`docs/MedSwin.tex`](MedSwin.tex)  
API contract: [`docs/ENDPOINTS.md`](ENDPOINTS.md)  
System-level TREC harness: [`eval/README.md`](../eval/README.md)

---

## 1. Why this baseline exists

MedSwin is not “RAG with more features.” It changes the decision the system is allowed to make:

- Generic RAG asks: *Which passages should be stuffed into the prompt?*
- MedSwin asks: *Is the available evidence enough to support a clinical answer?*

To measure whether that design is doing work, the control must be the textbook naive pipeline:

```text
query → embed → vector search (top-K) → generate
```

No BM25, no reranker, no score calibration, no fusion, no utility packing, no multi-agent claims, no evidence-sufficiency gate, no retrieve-more loop.

If full MedSwin cannot beat this control on evidence recall, groundedness, unsafe omission, or unjustified generation, the extra machinery is not earning its complexity.

---

## 2. What the system does today

### 2.1 Shared runtime (reused by both pipelines)

These services are **the same objects / same data** for naive-RAG and full MedSwin. That is the fairness contract.

| Shared piece | Module | Role |
| --- | --- | --- |
| Settings / env | `app/core/config.py` | Ports, Mongo, model URLs, `NAIVE_TOP_K` |
| Mongo corpus | `app/core/database.py`, `app/repositories/chunks.py` | Chunks + documents, `org_id` scoped |
| Ingest + chunking | `POST /api/v1/medswin/ingest`, `app/medswin/chunking.py` | Same passages in the store |
| Embedding client | `app/services/adapters/embedding.py` | Same vector space as `settings.active_embedding_url()` |
| Dense ANN | `app/retrieval/dense.py` → `app/indexes/hybrid.py` | HNSW ∪ FAISS/IVF over the same index files |
| Retrieval filters | `app/retrieval/filters.py` | Same `org_id`, optional `patient_id` / `source_policy` |
| LLM client | `app/services/adapters/llm.py` | Same supervisor / cloud chat model |
| Sessions + traces | `app/repositories/sessions.py`, `app/repositories/traces.py` | Both writes are queryable via `/medswin/traces/{id}` |
| Citation object | `app/services/medswin/governance.py` `build_citation` | Same citation shape for eval |
| Evidence bundle schema | `app/retrieval/hybrid.py` `build_bundle` | Naive reuses the **schema builder only** |

### 2.2 Full MedSwin (what naive-RAG must not call)

Implemented in `app/medswin/orchestrator.py`:

```text
QueryNormalizer
  → retrieve-more loop
      HybridRetriever.retrieve   (dense ∪ BM25, source-balanced probes, hints)
      HybridRetriever.rerank     (pointwise rerank + Platt/T calibration)
      MAC agents (EMR / Guideline / Safety / Quality / Critic)
      claim ledger + contradiction adjudication
      fusion + budgeted utility + packing
      EvidenceGate
        ACCEPT          → SynthesisAgent (citation-hardened CDS)
        RETRIEVE_MORE   → expand query / route specialist / loop
        INSUFFICIENT    → bounded abstention
```

Primary modules: `app/medswin/normalize.py`, `app/retrieval/hybrid.py`, `app/retrieval/lexical.py`, `app/services/adapters/reranker.py`, `app/scoring/*`, `app/agents/*`, `app/medswin/gate.py`, `app/medswin/abstain.py`.

### 2.3 Naive-RAG (the control)

Implemented in `app/medswin/naive.py`, served at `POST /api/v1/naive/chat`.

```text
raw query
  → EmbeddingClient.embed([query])
  → DenseRetriever.retrieve(k=NAIVE_TOP_K)
      if ANN empty and NAIVE_ENABLE_MONGO_FALLBACK
          → capped Mongo cosine scan (local-dev only)
  → take top-K by dense_score
  → one LLMClient.call_llm  (“answer from these passages; if weak, still answer”)
  → always generate (ungated)
```

Explicitly unused:

- `LexicalRetriever` / BM25
- `RerankerClient`
- `CalibrationStore`
- `compute_fusion_scores` / `select_bundle` / facet coverage
- `QueryNormalizer`, patient synopsis injection
- all MAC agents and `SynthesisAgent`
- `EvidenceGate` / retrieve-more / `expand_query`

Naive **always generates**, including when retrieval is empty (parametric fallback). That is a feature of the control: MedSwin’s abstention can only look valuable if the baseline is allowed to answer unsafely.

---

## 3. Fairness contract

Hold these constant across a comparison run:

1. **Same corpus.** Ingest once through `POST /api/v1/medswin/ingest`. Do not build a second chunk store for naive-RAG.
2. **Same embedding space.** Both call `EmbeddingClient(settings.active_embedding_url())`.
3. **Same ANN files.** `HNSW_INDEX_PATH` / `FAISS_INDEX_PATH` built from that corpus.
4. **Same `org_id`.** Default local org is `demo-org`. Benchmark org is `bench-org`.
5. **Same LLM.** Naive uses `SUPERVISOR_URL` / `CLOUD_MODEL`. MedSwin specialist agents may call the same model on other URLs; in `CLOUD_MODE` they all collapse to one cloud chat deployment.
6. **Same question text.** Do not paste the patient note into the naive prompt unless you also set `include_patient_context_in_query` on the eval harness (off by default). Patient notes belong in EMR ingest.

What is allowed to differ (this is the treatment):

| Dimension | Naive-RAG | Full MedSwin |
| --- | --- | --- |
| Query rewrite | none | normalize + optional EMR synopsis |
| Candidate pool | dense top-K (`NAIVE_TOP_K`, default 5) | hybrid union, `CANDIDATE_K` / `CANDIDATE_K_PRIME` |
| Lexical | off | BM25 if `ENABLE_BM25` |
| Rerank / calibration | off | on |
| Agents | off | MAC claim specialists |
| Selection | raw dense rank + char cap | fusion + utility + packing |
| Generation gate | always generate | accept / retrieve-more / abstain |
| Answer style | generic RAG prompt | structured CDS synthesis |

Mongo cosine fallback (`retrieval_backend=mongo_cosine`) is **not** a fair published baseline. It exists so `./scripts/start-local.sh --prompt` still works before you build an ANN index. For any number you would write down, build the index first and confirm `retrieval_backend=ann`.

---

## 4. Local setup (manual, from a clean checkout)

All commands assume the repository root.

### 4.1 Prerequisites

- Python 3.11+ recommended
- Docker (only if Mongo is not already on `localhost:27017`)
- Either:
  - `CLOUD_MODE=true` plus Azure AI Foundry embedding + chat credentials in `.env`, or
  - a local OpenAI-compatible stack on `EMBEDDING_URL` and `SUPERVISOR_URL` (see `env.example`)

Naive-RAG does **not** need a reranker endpoint. Full MedSwin does, unless you accept degraded rerank traces.

### 4.2 One-time files

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
# edit .env: MONGODB_URL, CLOUD_* or local model URLs
```

### 4.3 Start the runtime

```bash
./scripts/start-local.sh
```

This script:

1. Reuses `.venv` or `venv`, or creates `.venv`
2. Loads `.env` if present
3. Installs `requirements.txt`
4. Pings Mongo; if down, starts `mongo:6.0` as `rag_mongodb`
5. Exports MedSwin paths (`MONGODB_DB=medswin`, index files under `./data`)
6. Starts `uvicorn app.main:app` on **port 8100** (not 8000 — 8000 is the supervisor LLM in `env.example`)

Leave this terminal running. Confirm:

```bash
curl -s http://127.0.0.1:8100/health
```

OpenAPI: [http://127.0.0.1:8100/docs](http://127.0.0.1:8100/docs)  
Clinician UI: [http://127.0.0.1:8100/app/](http://127.0.0.1:8100/app/)

### 4.4 Ingest at least one document

Both pipelines read the same chunks. `source_type` and `org_id` are query parameters:

```bash
curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=LIT&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{
    "doc_id": "sample-lit-metformin",
    "title": "Metformin renal guidance (sample)",
    "text": "Recommendations\n\nMetformin may continue when eGFR is at or above the labelled threshold. Reassess dose between 45 and 59. Stop below the contraindicated threshold and watch for lactic acidosis."
  }]'
```

If you omit `chunks`, ingest runs `app/medswin/chunking.py` (section-aware ~350–450 tokens).

### 4.5 Build the ANN index (required for a fair comparison)

```bash
curl -s http://127.0.0.1:8100/api/v1/storage/index/build \
  -H 'Content-Type: application/json' \
  -d '{"force_rebuild": true, "org_id": "demo-org"}'
```

Then check:

```bash
curl -s 'http://127.0.0.1:8100/api/v1/storage/stats?org_id=demo-org'
```

You want `index_exists=true` and a non-zero chunk count for `demo-org`.

---

## 5. Prompting from the terminal

`scripts/start-local.sh` can either start the server **or** attach a prompt client.

### 5.1 Interactive REPL (choose full / naive / both each turn)

With the server already running in another terminal:

```bash
./scripts/start-local.sh --prompt
```

If the API is down, `--prompt` starts it in the background, waits for `/health`, then opens the REPL. On exit it stops only the server **it** started.

You will be asked:

```text
Pipeline [full/naive/both] (both):
Question:
patient_id (optional):
```

- `full`  → `POST /api/v1/medswin/chat`
- `naive` → `POST /api/v1/naive/chat`
- `both`  → `POST /api/v1/naive/compare`

### 5.2 One-shot question

```bash
./scripts/start-local.sh --mode naive --question "Can this patient continue metformin after the latest renal-function result?"
./scripts/start-local.sh --mode full  --question "Can this patient continue metformin after the latest renal-function result?" --patient-id patient-42
./scripts/start-local.sh --mode both  --question "Can this patient continue metformin after the latest renal-function result?" --org-id demo-org
```

### 5.3 Call the Python client directly

```bash
source .venv/bin/activate
python -m app.cli.prompt --mode both --question "What first-line therapy is appropriate?"
python -m app.cli.prompt --json --mode naive --question "What first-line therapy is appropriate?"
```

Useful flags: `--base-url`, `--org-id`, `--user-id`, `--patient-id`, `--top-k`, `--timeout`, `--json`.

### 5.4 How to read a `both` result

The terminal prints three blocks:

1. **NAIVE RAG** — ungated answer, dense top-K passages, `timing_ms`
2. **FULL MEDSWIN** — gated/synthesized answer, selected bundle, policy action
3. **DIFF**
   - `jaccard` of chunk IDs
   - `overlap_chunk_ids` / `naive_only` / `medswin_only`
   - `medswin_abstained` — the design signal (naive almost never abstains)
   - wall-clock `timing_ms`

If naive shows `retrieval_backend=mongo_cosine` or `empty`, stop and build the index before treating the run as a benchmark.

---

## 6. HTTP API

Base: `http://127.0.0.1:8100`

### 6.1 Naive chat

`POST /api/v1/naive/chat`

```json
{
  "query": "Can this patient continue metformin after the latest renal-function result?",
  "user_id": "clinician-1",
  "org_id": "demo-org",
  "patient_id": "patient-42",
  "top_k": 5
}
```

Response is a `ChatResponse` with:

- `pipeline`: `"naive_rag"`
- `retrieval_backend`: `"ann"` | `"mongo_cosine"` | `"empty"` | `"error"`
- `timing_ms`: `{embed, retrieve, generate, total}`
- `policy_decision.passed`: always `true` (ungated)
- `uncertainty_level`: `"ungated"`
- `rerank_traces`: empty
- `evidence_bundle.passages`: dense top-K

### 6.2 Side-by-side

`POST /api/v1/naive/compare`

Same body as naive chat. Response:

```json
{
  "query": "...",
  "naive": { "pipeline": "naive_rag", "...": "..." },
  "medswin": { "pipeline": "medswin", "...": "..." },
  "diff": {
    "jaccard": 0.0,
    "overlap_chunk_ids": [],
    "naive_only_chunk_ids": [],
    "medswin_only_chunk_ids": [],
    "medswin_abstained": true,
    "timing_ms": { "naive": 1200.0, "medswin": 8400.0 }
  }
}
```

Full MedSwin remains `POST /api/v1/medswin/chat`. Traces for **both** pipelines are stored and can be fetched with `GET /api/v1/medswin/traces/{trace_id}?org_id=demo-org&include_details=true`.

---

## 7. Batch evaluation (same cases, two pipelines)

The existing TREC harness in `eval/` now accepts `pipeline`.

Start MedSwin on `:8100`, ingest the PMC / case corpus as in `eval/README.md`, then start the eval service **or** use the compare script.

### 7.1 Eval service

```bash
cd eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --port 8200
```

```bash
# Full system
curl -s http://127.0.0.1:8200/api/run \
  -H 'Content-Type: application/json' \
  -d '{"cases_path":"data/sample/cases.jsonl","max_cases":2,"pipeline":"medswin"}'

# Naive control
curl -s http://127.0.0.1:8200/api/run \
  -H 'Content-Type: application/json' \
  -d '{"cases_path":"data/sample/cases.jsonl","max_cases":2,"pipeline":"naive_rag"}'

# Both, plus a comparison sidecar
curl -s http://127.0.0.1:8200/api/run \
  -H 'Content-Type: application/json' \
  -d '{"cases_path":"data/sample/cases.jsonl","max_cases":2,"pipeline":"both"}'
```

From the **repository root**, the smoke file is `eval/data/sample/cases.jsonl`. If you `cd eval`, use `data/sample/cases.jsonl`.

### 7.2 One script, no eval UI

From the repository root, with MedSwin healthy and a corpus already indexed for `BENCHMARK_ORG_ID` (default `bench-org`):

```bash
python3 eval/scripts/run_pipeline_compare.py \
  --cases-path eval/data/sample/cases.jsonl \
  --max-cases 2
```

Writes under `RUN_STORE_DIR` (default `/tmp/medswin-audits`):

- `{medswin_run_id}.json`
- `{naive_run_id}.json`
- `{medswin_run_id}.comparison.json` with `delta_medswin_minus_naive` and per-case Jaccard / MSAS deltas

For publication numbers, point `--cases-path` at the TREC JSONL produced by `eval/scripts/prepare_trec_cds.py` and use the judged PMC ingest in `eval/README.md`. Do not publish smoke-file scores.

### 7.3 What the harness does not punish on naive runs

`eval/app/runner.py` skips MedSwin-only architecture errors when `pipeline=naive_rag` (missing sufficiency checks, missing agent messages, etc.). Those gaps are the baseline, not a crash.

Shared metrics still apply and are the ones to compare:

| Metric | What it tells you |
| --- | --- |
| `evidence_doc_recall` | Did top-K / selected bundle recover judged docs? |
| `citation_precision` | Are cited docs in the qrel set? |
| `facet_recall` / `critical_facet_recall` | Did selected docs cover gold clinical facets? |
| `sufficiency_decision_score` | Naive always “passes”; if critical facets are missing this score goes to 0 — that is the ungated-generation penalty |
| `unsafe_omission_penalty` | `1 - critical_facet_recall` |
| `groundedness_proxy` | Citation / ledger support (replace with clinician adjudication for the paper) |
| `trace_completeness` | Naive will score lower; that is expected |
| `MSAS` | Composite; interpret per-term, not as a single trophy number |
| `policy_pass_rate` | Naive ≈ 1.0; MedSwin < 1.0 can be correct abstention |
| chunk Jaccard (compare) | Did the two pipelines even look at the same evidence? |

---

## 8. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NAIVE_TOP_K` | `5` | Dense hits stuffed into the naive prompt |
| `NAIVE_MAX_CONTEXT_CHARS` | `8000` | Hard cap so naive cannot dump the whole scan |
| `NAIVE_ENABLE_MONGO_FALLBACK` | `true` | Cosine scan if ANN returns nothing |
| `NAIVE_MONGO_SCAN_LIMIT` | `4000` | Max chunks loaded for that fallback |
| `DEFAULT_TOP_K` | `5` | Legacy retrieval default; keep aligned with `NAIVE_TOP_K` for apples-to-apples |
| `CANDIDATE_K` / `CANDIDATE_K_PRIME` | `80` / `120` | MedSwin only |
| `ENABLE_BM25` | `true` | MedSwin only |
| `MAX_RETRIEVE_LOOPS` | `3` | MedSwin only |

Set them in `.env`. `env.example` lists the naive keys.

---

## 9. Code map

```text
app/medswin/naive.py                 NaiveRAGOrchestrator + compare_responses()
app/api/v1/endpoints/naive.py        POST /naive/chat, POST /naive/compare
app/cli/prompt.py                    python -m app.cli.prompt
scripts/start-local.sh               env + Mongo + API + optional prompt
eval/app/client.py                   pipeline=naive_rag uses /api/v1/naive/chat
eval/app/runner.py                   pipeline=medswin|naive_rag|both
eval/app/compare.py                  aggregate deltas
eval/scripts/run_pipeline_compare.py CLI for pipeline=both
tests/test_naive_rag.py              control-pipeline unit tests
```

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `API health check failed` | uvicorn not up, wrong port | `./scripts/start-local.sh` then `curl /health`. Default port is **8100**. |
| Naive `retrieval_backend=empty` | no chunks for `org_id`, or embeddings missing | ingest with that `org_id`. Ingest now attaches embeddings in every mode when the embedding client (or local ModelManager) works |
| Naive answer mentions “0 embeddings” / CLI exit 1 | chunks stored without vectors | `POST /api/v1/storage/embeddings/refresh` then `POST /api/v1/storage/index/build` |
| Chat hangs for minutes then fails | old rate-limiter treated connection-refused as a 429 | connection errors now fail after two short retries; upgrade to this revision |
| `GET /api/v1/naive/ready` | preflight | `mongo`, `chunk_count`, `embedded_count`, `index_exists` |
| Naive `retrieval_backend=mongo_cosine` | ANN not built or mapping empty | `POST /api/v1/storage/index/build` |
| Full MedSwin 500 / degraded rerank | no `RERANKER_URL` / cloud reranker | expected for naive; fix before publishing full-system numbers |
| Both answers ignore the patient | note never ingested as EMR | `ingest?source_type=EMR` with `patient_id`, or `--patient-id` after ingest |
| Compare is very slow | full MAC loop + retrieve-more | expected; raise `--timeout` (CLI default 300s) |
| Eval qrel coverage error | smoke cases have gold IDs that are not in `bench-org` | ingest those docs into the benchmark org, or use TREC prep scripts |
| Script uses port 8000 | old habit | supervisor LLM is 8000; the API is 8100 |

---

## 11. Reproducibility checklist

Copy this and tick it before you treat a run as real:

1. Record git SHA: `git rev-parse HEAD`
2. Record `.env` keys that change behaviour (`CLOUD_MODE`, embedding model, `NAIVE_TOP_K`, `ENABLE_BM25`, sufficiency thresholds). Do **not** commit secrets.
3. Record `org_id` and whether the ANN index was rebuilt (`force_rebuild`)
4. Confirm `/health` and `storage/stats` for that org
5. Confirm naive `retrieval_backend=ann` on a probe question
6. Run the **same** `cases_path` / question twice: `pipeline=naive_rag` then `pipeline=medswin`, or one `pipeline=both`
7. Save the three JSON artefacts (naive audit, MedSwin audit, comparison)
8. Report per-metric deltas, abstention rate, and chunk Jaccard — not MSAS alone
9. For TREC, follow `eval/README.md` judged-pool ingest and qrel coverage gates

---

## 12. Tests

From the repository root:

```bash
python3 -m pytest tests/test_naive_rag.py tests/test_eval_harness.py -q
```

`test_naive_rag.py` asserts: dense top-K only, no reranker, generation with an empty pool, and the compare helper’s Jaccard / abstention fields.
