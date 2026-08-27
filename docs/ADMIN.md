# MedSwin administrator runbook

Read this page first if you need to **configure, start, ask, compare, or evaluate** the live system. Other files in this folder are specialist manuals. This page is the governed day-one path.

Executable source of truth: `app/medswin/`, `app/cli/operator.py`, `scripts/start-local.sh`.  
Do **not** start from `lab/`, root `docker-compose.yml`, `Dockerfile`, or `aws/deploy.sh` unless you are maintaining those legacy artefacts (see §12).

---

## 0. What you are running

MedSwin is an evidence-gated multi-agent clinical decision-support runtime. Top-K retrieval does **not** authorize an answer. After hybrid retrieve, rerank, and specialist claims, an evidence-sufficiency gate either synthesizes a grounded CDS answer or returns a bounded abstention.

A **naive-RAG control** lives on the same process so you can measure whether that design is doing work:

| Pipeline | Route | What it does |
| --- | --- | --- |
| Full MedSwin | `POST /api/v1/medswin/chat` | Normalize → hybrid (HNSW ∪ IVF ∪ BM25) → rerank → MAC agents → gate → synthesize or abstain |
| Naive-RAG | `POST /api/v1/naive/chat` | Embed → dense top-K → one generate. Ungated, except infrastructure gaps (see §7) |
| Compare | `POST /api/v1/naive/compare` | Same body through both; returns answers + chunk Jaccard + abstention |

Both pipelines share: Mongo corpus, embeddings, ANN files, `org_id` / patient filters, and the same LLM backend. That is the fairness contract. Details: [NAIVE_RAG.md](NAIVE_RAG.md).

Publication evaluation is [PAPER_EVAL.md](PAPER_EVAL.md). It is not a second web portal.

---

## 1. Identities, ports, files

| Name | Value | Used by |
| --- | --- | --- |
| API listen | **8100** (`APP_PORT`) | Chat, ingest, traces, clinician UI, dashboard, OpenAPI |
| Paper-eval API | **8110** (`FULL_EVAL_API_PORT`) | Isolated T3/T4 process; T1 exporter does not call `/chat` |
| Supervisor LLM | 8000 (`SUPERVISOR_URL`) | Naive generate + MedSwin supervisor / synthesis |
| Specialist LLMs | 8001–8003 | Optional; `CLOUD_MODE=true` collapses them |
| Reranker | 8004 | Required for a fair **full** MedSwin compare |
| Embeddings HTTP | 8005 | Optional; local mode can use `ModelManager` |
| Mongo URL | `mongodb://localhost:27017` | |
| Mongo database | **`medswin`** (`MONGODB_DB`) | Not `medical_rag_db` |
| Local ask org | **`demo-org`** | Clinician UI, `ask`, `index` |
| Paper-eval org | **`bench-org`** (`BENCHMARK_ORG_ID`) | `paper-eval` — **not** `--org-id` |
| Ask user | `clinician-1` | |
| Eval user | `bench-user` | |
| Auto Mongo container | **`medswin-mongodb`** (`mongo:7.0`) | Started by `start-local.sh` if ping fails |
| API PID / log | `logs/medswin-api.pid`, `logs/medswin-api.log` | Only if this operator started uvicorn |
| Local 7B PID / log | `logs/medswin-llm.pid`, `logs/medswin-llm.log` | Optional T3 generator |
| Ordinary ANN | `./data/hnsw_index.bin` + mapping; optional FAISS | Chat path |
| Paper-eval artifacts | `FULL_EVAL_DATA_DIR` default `./data/full-trec-benchmark` | Isolated HNSW / BM25 |
| Calibration | `./data/calibration/rerank.json`, `agents.json` | Identity `{b:0,T:1}` until you fit Platt |

`q` in the console does **not** stop uvicorn. Use `./scripts/start-local.sh stop`.

---

## 2. Prerequisites

- Python 3.11+ recommended (3.12 works).
- Docker only if Mongo is not already on `localhost:27017`.
- One of:
  - **Cloud:** `CLOUD_MODE=true` plus Azure AI Foundry endpoint + API key in `.env`.
  - **Local models:** OpenAI-compatible `SUPERVISOR_URL` and either `EMBEDDING_URL` or weights under `models/MedEmbed-large-v0.1`.
- Full MedSwin also needs a reranker (`RERANKER_URL` or `models/bge-reranker-v2-m3`). Naive-RAG does not.
- Repository root as the working directory for every command below.
- Paper-eval T1 scoring needs official NIST files and `trec_eval` (see [PAPER_EVAL.md](PAPER_EVAL.md)).

---

## 3. Configure `.env`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

Never commit `.env`. `env.example` is the key catalogue. Minimum you must set:

### Local HTTP models

```text
CLOUD_MODE=false
APP_PORT=8100
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=medswin
SUPERVISOR_URL=http://localhost:8000/v1/chat/completions
EMBEDDING_URL=http://localhost:8005/embeddings
RERANKER_URL=http://localhost:8004/rerank
```

If `EMBEDDING_URL` is down, ingest / naive fall back to a loaded `ModelManager` when `EMBEDDING_MODEL_PATH` exists.

### Cloud (Azure AI Foundry)

```text
CLOUD_MODE=true
AZURE_AI_FOUNDRY_ENDPOINT=https://<resource>.services.ai.azure.com
AZURE_AI_FOUNDRY_API_KEY=<secret>
CLOUD_MODEL=gpt-5.4
CLOUD_EMBEDDING=embed-v-4-0
CLOUD_RERANKER=Cohere-rerank-v4.0-fast
```

In cloud mode, startup **does not** download local HF weights and **does not** auto-refresh embeddings. After ingest, call `POST /api/v1/storage/embeddings/refresh` or `./scripts/start-local.sh index`.

### Paper-eval (same `.env` is fine)

```text
MEDSWIN_BASE_URL=http://127.0.0.1:8100
BENCHMARK_ORG_ID=bench-org
BENCHMARK_USER_ID=bench-user
TREC_EVAL_BIN=
NIST_DIR=
FULL_EVAL_API_PORT=8110
FULL_EVAL_DATA_DIR=./data/full-trec-benchmark
EVAL_WARMUP_ON_START=false
```

### Knobs you will actually change

| Key | Default | When to touch it |
| --- | --- | --- |
| `NAIVE_TOP_K` | 5 | Fair local compare |
| `ENABLE_BM25` | true | MedSwin only |
| `MAX_RETRIEVE_LOOPS` | 3 | Full-system latency |
| `SUFF_CRITICAL_FACET_THRESHOLD` | 0.78 | How often MedSwin abstains |
| `CLOUD_EMBED_BATCH_SIZE` / `CLOUD_EMBED_BATCH_DELAY_S` | 64 / 60 | Cloud quota |
| `DEBUG` | false in `env.example` | `start-local.sh` **forces `DEBUG=true`** during bootstrap unless you already exported it |

Full list: [`env.example`](../env.example). Invalid sufficiency thresholds fail API startup.

---

## 4. First-run sequence (copy this)

```bash
# 1. Config
cp env.example .env
# edit secrets / CLOUD_MODE / model URLs

# 2. Start (TTY → operator console; starts Mongo + API if needed)
./scripts/start-local.sh

# 3. Confirm
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1:8100/api/v1/naive/ready

# 4. Ingest at least one document into demo-org
curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=LIT&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{"doc_id":"sample-lit-metformin","title":"Metformin renal guidance","text":"Recommendations\n\nMetformin may continue when eGFR is at or above the labelled threshold."}]'

# 5. Embed + build ANN
./scripts/start-local.sh index --org-id demo-org

# 6. Ask both pipelines
./scripts/start-local.sh ask --mode both --question "Can this patient continue metformin after the latest renal-function result?"

# 7. Open the clinician UI (same three pipelines)
./scripts/start-local.sh open clinician
```

You want naive `retrieval_backend=ann` on a probe question. `mongo_cosine` or `empty` means the index is not ready — do not treat that as a design result.

---

## 5. Operator commands

Entry: [`scripts/start-local.sh`](../scripts/start-local.sh) → [`app/cli/operator.py`](../app/cli/operator.py) for local ask/status. Paper numbers go through `paper-eval` in the shell.

```text
./scripts/start-local.sh [command] [options]
```

| Command | What happens |
| --- | --- |
| `console` | Default in a TTY. Bootstrap, print URLs, command menu |
| `serve` | Default when stdin is not a TTY. Foreground `uvicorn app.main:app` |
| `up` | Start API, print status, optionally open browsers |
| `ask` | One question or REPL. `--mode full\|naive\|both` |
| `paper-eval` | Official TREC T1 (+ optional T3/T4). See [PAPER_EVAL.md](PAPER_EVAL.md) |
| `eval` | Removed; exits 1 |
| `open` | `clinician` \| `dashboard` \| `docs` \| `all` |
| `status` | Health, naive ready, storage stats, paper-eval NIST/runs |
| `index` | `POST /storage/embeddings/refresh` then `POST /storage/index/build` |
| `stop` | SIGTERM PIDs this operator wrote under `logs/` |

Aliases: `--prompt` / `--question` → `ask`.  
Python: `python -m app.cli.operator console` after `source .venv/bin/activate`.

Typical:

```bash
./scripts/start-local.sh
./scripts/start-local.sh up --open
./scripts/start-local.sh ask --mode both --question "What first-line therapy is appropriate?" --patient-id patient-7
./scripts/start-local.sh paper-eval
./scripts/start-local.sh index --org-id demo-org
./scripts/start-local.sh status
./scripts/start-local.sh stop
```

Console menu: `1` full, `2` naive, `3` both, `4–5` portals, `6` paper-eval status, `7` index, `8` status, `q` quit (servers stay up). A line of three or more words (or a trailing `?`) is treated as a question in the current mode.

`--port` wins over `APP_PORT` from `.env`. Full flag table: [OPERATOR.md](OPERATOR.md).

Python cache only (does not delete Mongo or ANN):

```bash
./scripts/delete-cache.sh
```

---

## 6. Web surfaces

After the API is up, all of these share the **same** Mongo and the **same** uvicorn process.

| Surface | URL | Purpose |
| --- | --- | --- |
| Clinician CDS | http://127.0.0.1:8100/app/ | Ask full / naive / both; inspect answer, passages, facets, contradictions, traces |
| Ops dashboard | http://127.0.0.1:8100/api/v1/dashboard/ | HF ingest, model download, service status |
| OpenAPI | http://127.0.0.1:8100/docs | Live schemas |
| Health | http://127.0.0.1:8100/health | Liveness; cloud adds embedding-space + refresh |
| Naive preflight | http://127.0.0.1:8100/api/v1/naive/ready | Mongo, chunk vs embedded counts, index file |

### Clinician UI (`/app/`)

Served from `web/public/index.html` unless you build the React SPA (`cd web && npm run build` → `web/dist/`, which FastAPI prefers).

| Field | In the form? | Notes |
| --- | --- | --- |
| Pipeline | yes | `full` → `/medswin/chat`, `naive` → `/naive/chat`, `both` → `/naive/compare` |
| Question | yes | Required |
| `patient_id` | yes | Optional EMR scope |
| `org_id` | yes | Default `demo-org` |
| `user_id` | yes | Default `clinician-1` |
| Naive `top_k` | yes | 1–20; ignored by full MedSwin |
| `session_id` | **no** | Use curl / operator if you need a sticky session |
| `constraints` | **no** | `source_policy`, `min_evidence_grade`, `disable_gate`, `disable_mac` — use HTTP |

Compare view: side-by-side answers + `diff` (Jaccard, overlap, `medswin_abstained`). Trace fetch (`GET /medswin/traces/{id}?include_details=true`) is on the single-pipeline path.

Vite dev (optional): `cd web && npm run dev` on `:5173`, proxies `/api` → `:8100`. Override with `VITE_API_BASE`.

---

## 7. Naive-RAG behaviour (so you do not misread a failed setup)

Ungated **generation** is the control — but only when the corpus is usable.

| Situation | LLM called? | `retrieval_backend` / `degraded_mode` |
| --- | --- | --- |
| ANN returned hits | yes | `ann` |
| ANN empty, mongo cosine found hits | yes | `mongo_cosine` (local-dev only; not publishable) |
| No chunks for this org | yes (parametric) | `empty` — this **is** the ungated control |
| Chunks exist, **0 embeddings** | **no** | `degraded_mode.no_embeddings` — fix with `index` |
| Vector dimension mismatch | **no** | `dim_mismatch` / `empty_index` |
| Exception | no | `error` |

One-shot `ask` exits **1** on `no_embeddings` or `error`. The console stays open so you can `index` and retry.

---

## 8. Corpus and index

Honest compares require the same embedded ANN.

```bash
curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=LIT&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{"doc_id":"sample-lit-metformin","title":"…","text":"…"}]'

curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=EMR&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{"doc_id":"note-42","patient_id":"patient-42","title":"Admission","text":"…"}]'

./scripts/start-local.sh index --org-id demo-org
curl -s 'http://127.0.0.1:8100/api/v1/storage/stats?org_id=demo-org'
curl -s http://127.0.0.1:8100/api/v1/naive/ready
```

Ingest always **tries** to attach the active embedding space. Cloud: embed failure is fatal. Local: warning, text still stored — then you must refresh.

You want `index_exists=true`, `embedded_count > 0`, matching `index_manifest.org_id` / embedding space. Chat queries **HNSW ∪ FAISS IVF** (union). BallTree exists in `app/core/indexing/` but is **not** on the MedSwin chat path. See [INDEXING.md](INDEXING.md).

Reset a benchmark tenant before a fresh TREC ingest:

```bash
curl -s http://127.0.0.1:8100/api/v1/storage/benchmark/reset \
  -H 'Content-Type: application/json' \
  -d '{"org_id":"bench-org","remove_indexes":true}'
```

---

## 9. Evaluation

Publication path: [PAPER_EVAL.md](PAPER_EVAL.md).

```bash
./scripts/start-local.sh warmup
./scripts/start-local.sh paper-eval
./scripts/start-local.sh paper-eval --pipeline both --generator cloud --stage t3
```

T1 is LIT-only official IR (note + type question → 1000 PMCIDs → NIST `sample_eval` / `trec_eval`). It does **not** call `/chat`.  
T3 is the product path: note ingested as EMR, query = type question, frozen Foundry GPT.  
T4 is automatic `full` / `-gate` / `-MAC` rates. Human T4 is not confirmatory.

`eval`, `eval --run`, and `:8200` are gone. Do not publish homemade MSAS composites. T1 does not authorize a “gated multi-agent CDSS is SOTA” sentence.

---

## 10. HTTP cheat sheet

Base `http://127.0.0.1:8100`. Full contract: [ENDPOINTS.md](ENDPOINTS.md).

```bash
curl -s http://127.0.0.1:8100/api/v1/medswin/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Can metformin continue?","user_id":"clinician-1","org_id":"demo-org","patient_id":"patient-42"}'

curl -s http://127.0.0.1:8100/api/v1/naive/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Can metformin continue?","user_id":"clinician-1","org_id":"demo-org","top_k":5}'

curl -s 'http://127.0.0.1:8100/api/v1/medswin/traces/TRACE_ID?org_id=demo-org&include_details=true'
```

Insufficient evidence is HTTP **200** with `policy_decision.passed=false`. That is success of the gate, not an API error.

---

## 11. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Health check failed | `logs/medswin-api.log`; Mongo ping; `.env` model URLs. Port is **8100**, not 8000 |
| Console cannot import `app.cli.operator` | Run from repo root; `source .venv/bin/activate` |
| Port in use | `status`, reuse it, or `--port 8101` |
| Naive “0 embeddings” / CLI exit 1 | `index --org-id <org>` |
| `retrieval_backend=mongo_cosine` | Build ANN; do not publish |
| Full MedSwin degraded rerank | Start reranker or set cloud reranker. Naive does not need it |
| `eval` exits 1 | Use `paper-eval` |
| paper-eval NIST / `trec_eval` missing | `./scripts/start-local.sh warmup` |
| Answers ignore the patient | EMR ingest with matching `patient_id` + `org_id` |
| Compare is slow | Full MAC + up to 3 retrieve-more loops; CLI timeout default 300s |
| Browsers did not open | `open clinician`, or paste URLs from `status` |
| Followed Docker Compose and hit port 8000 / `medical_rag_db` | That file is **legacy**. Use this runbook |

---

## 12. What not to follow (legacy / research)

| Path | Status |
| --- | --- |
| `./scripts/start-local.sh` | **Canonical** local operator |
| `lab/` | Pre-production HPC modules. Not the FastAPI path |
| Root `docker-compose.yml` | Legacy: API **8000**, DB `medical_rag_db`, optional nginx. Does not match this runbook |
| `Dockerfile` | Legacy `EXPOSE 8000` / uvicorn `--port 8000` |
| `aws/deploy.sh` | Deploys that compose to a hardcoded EC2 host. Not the operator CLI or clinician UI |
| `eval/` | Removed. Do not restore the MSAS portal |
| `docs/MedSwin.tex` | Architecture paper; **not** a tracked runtime contract |

If you must use Compose, you are on a different contract: set `APP_PORT` / `MONGODB_DB=medswin` yourself and do not mix those numbers with this runbook. Paper-eval prefers the Compose `mongodb` service (`medswin-mongodb` / `mongo:7.0`).

---

## 13. Governance checklist (before you treat a run as real)

1. Record `git rev-parse HEAD`
2. Record non-secret `.env` keys that change behaviour (`CLOUD_MODE`, embedding model, `NAIVE_TOP_K`, `ENABLE_BM25`, sufficiency thresholds, `FULL_EVAL_*`)
3. Record `org_id` (demo vs bench) and whether ANN was rebuilt
4. `GET /health` and `GET /api/v1/naive/ready` — `embedded_count > 0`, `index_exists`
5. Probe question shows naive `retrieval_backend=ann` for local compares
6. For T1: official NIST SHA256 pins, `validate.py`, then `sample_eval` + `trec_eval -q -c -M1000`
7. For T3: note as EMR, type question as query, Foundry GPT freeze, persisted packs
8. Report T1 as a retriever table. Gate/MAC claims live on T3/T4
9. Follow [PAPER_EVAL.md](PAPER_EVAL.md)

---

## 14. Tests

```bash
python3 -m pytest tests/test_operator_cli.py tests/test_naive_rag.py tests/test_paper_eval_cli.py -q
python3 -m pytest tests/test_trec_cds2016.py tests/test_expert_protocol.py tests/test_full_eval_contract.py -q
python3 -m pytest tests/test_medswin_policy.py tests/test_medswin_governance.py tests/test_medswin_retrieval.py -q
```

---

## 15. Specialist manuals

| Document | Open it when |
| --- | --- |
| [OPERATOR.md](OPERATOR.md) | Console flags and menu |
| [PAPER_EVAL.md](PAPER_EVAL.md) | Official TREC T1–T4 |
| [NAIVE_RAG.md](NAIVE_RAG.md) | Fairness contract and local compare |
| [ENDPOINTS.md](ENDPOINTS.md) | Every HTTP route |
| [MEDSWIN.md](MEDSWIN.md) | MAC, gate math, traces, fail-open |
| [INDEXING.md](INDEXING.md) | Embed, refresh, HNSW ∪ IVF |
| [../README.md](../README.md) | Product diagrams and data model |
