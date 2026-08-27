# Local operator

Day-to-day manual for one machine: start the API, ask full MedSwin or the naive-RAG control, open the clinician UI, rebuild the index, and inspect paper-eval status.

The entry point is [`scripts/start-local.sh`](../scripts/start-local.sh). Asking a question reuses [`app/cli/prompt.py`](../app/cli/prompt.py). Publication numbers are **not** an operator submenu — they are `paper-eval` (see [PAPER_EVAL.md](PAPER_EVAL.md)).

Day-one runbook: [ADMIN.md](ADMIN.md). Related: [NAIVE_RAG.md](NAIVE_RAG.md), [ENDPOINTS.md](ENDPOINTS.md).

---

## 1. Surfaces

These share the same Mongo corpus and the same API process. There is no eval portal.

| Surface | URL | What you do there |
| --- | --- | --- |
| Clinician CDS | http://127.0.0.1:8100/app/ | Ask **full MedSwin**, **naive-RAG**, or **both**. Inspect answer, passages, facets, contradictions, traces |
| Ops dashboard | http://127.0.0.1:8100/api/v1/dashboard/ | Corpus / Hugging Face ingest / model download |
| OpenAPI | http://127.0.0.1:8100/docs | Live request schemas |
| Health | http://127.0.0.1:8100/health | Process liveness and loaded models |
| Naive preflight | http://127.0.0.1:8100/api/v1/naive/ready | Mongo, chunk count, embedding count, index file |

The clinician page at `/app/` is the zero-build UI in [`web/public/index.html`](../web/public/index.html). After `npm run build` in `web/`, FastAPI prefers `web/dist/`.

Form fields: pipeline (`full` / `naive` / `both`), question, `patient_id`, `org_id` (default `demo-org`), `user_id` (default `clinician-1`), naive `top_k` (1–20). The UI does **not** send `session_id` or `constraints` — use curl. Nav links: clinician, ops dashboard, OpenAPI.

---

## 2. Commands

```text
./scripts/start-local.sh [command] [options]
```

| Command | Behaviour |
| --- | --- |
| `console` | Default in a TTY. Ensures venv / Mongo / API, prints portal URLs, opens a command menu |
| `serve` | Default when stdin is not a TTY. `uvicorn app.main:app` in the foreground |
| `up` | Start the API, print status, open the clinician and dashboard browsers |
| `ask` | One question or a REPL. Modes: `full`, `naive`, `both` |
| `paper-eval` | Official TREC CDS 2016 + optional T3/T4. See [PAPER_EVAL.md](PAPER_EVAL.md) |
| `warmup` | NIST files, `trec_eval`, `ir_datasets`; HF only with `--with-local-llm` |
| `eval` | Removed. Prints the replacement and exits 1 |
| `full-eval` | Deprecated alias for `paper-eval --stage all` |
| `open` | Open `clinician`, `dashboard`, `docs`, or `all` |
| `status` | API health, naive ready, storage stats, paper-eval NIST/runs/scores |
| `index` | `POST /storage/embeddings/refresh` then `POST /storage/index/build` |
| `stop` | SIGTERM the API / local-LLM PIDs this operator wrote under `logs/` |

Legacy flags still work:

```bash
./scripts/start-local.sh --prompt
./scripts/start-local.sh --mode both --question "Can metformin continue?"
```

Those are aliases for `ask`. Removed publication flags (`--with-eval`, `--eval-port`, `--run`, `--cases-path`, `--max-cases`) exit 1.

### Defaults

- TTY session → `console`
- Piped / CI session → `serve`
- API port `8100`, org `demo-org`, user `clinician-1`
- `--port` wins over `APP_PORT` from `.env`
- `EVAL_WARMUP_ON_START=false` — ordinary start does not download TREC

### Typical sessions

```bash
./scripts/start-local.sh
# then type a question, or: 3   (compare both)
# then: open clinician

./scripts/start-local.sh up --open
./scripts/start-local.sh ask --mode both --question "What first-line therapy is appropriate?" --patient-id patient-7
./scripts/start-local.sh paper-eval
./scripts/start-local.sh index --org-id demo-org
./scripts/start-local.sh stop
```

Equivalent Python (after `source .venv/bin/activate`):

```bash
python -m app.cli.operator console
python -m app.cli.operator ask --mode naive --question "..."
python -m app.cli.operator status --json
python -m app.cli.prompt --mode both --question "..."
```

---

## 3. Console menu

```text
  1  ask full          POST /api/v1/medswin/chat
  2  ask naive         POST /api/v1/naive/chat
  3  ask both          POST /api/v1/naive/compare
  4  open clinician
  5  open dashboard
  6  paper-eval        Last T1 run / score paths
  7  index             refresh embeddings + rebuild ANN
  8  status
  h  help
  q  quit              leaves servers running
```

You can also type a clinical question (three or more words, or a trailing `?`) and the console uses the current default mode (`mode full|naive|both` changes it).

`q` does **not** kill uvicorn. Use `stop` when you want the API down. PIDs live in:

```text
logs/medswin-api.pid
logs/medswin-api.log
logs/medswin-llm.pid
logs/medswin-llm.log
```

Only processes this operator started are stopped. An API you launched yourself with `serve` is left alone.

---

## 4. What the bootstrap does

`start-local.sh` (for commands other than `status` / `open` / `stop` / `eval`):

1. Finds `.venv` or `venv`, or creates `.venv`
2. Loads `.env` if present
3. Installs `requirements.txt` only if `fastapi`, `uvicorn`, `httpx`, or `pymongo` are missing
4. Pings `MONGODB_URL` (default `mongodb://localhost:27017`). If down, starts Compose Mongo `medswin-mongodb` / `mongo:7.0`
5. Forces `MONGODB_DB=medswin` unless you already exported another name
6. Exports `DEBUG=true` unless you already set `DEBUG`
7. Creates `models/`, `data/`, `logs/`, `storage/`
8. Warns if local HF weights are absent
9. Hands off to `python -m app.cli.operator …`, uvicorn, or `paper-eval`

Python bytecode only (safe; does not wipe Mongo or `data/*.bin`):

```bash
./scripts/delete-cache.sh
```

Mongo database name is **`medswin`**. Port **8000** in `env.example` is the supervisor LLM, not this API.

---

## 5. Asking a question

| Mode | Route | Meaning |
| --- | --- | --- |
| `full` | `POST /api/v1/medswin/chat` | Hybrid retrieve, rerank, MAC, sufficiency gate |
| `naive` | `POST /api/v1/naive/chat` | Embed → dense top-K → generate. Always answers |
| `both` | `POST /api/v1/naive/compare` | Same body through both; terminal prints a retrieval diff |

Shared request fields: `query`, `user_id`, `org_id`, optional `patient_id`, `session_id`, `constraints`. Naive also accepts `top_k` (default `NAIVE_TOP_K=5`).

Patient notes belong in EMR ingest, not pasted into the question. T3 packs ingest the official note as EMR and query with the type question only (`include_patient_context_in_query=false`).

If naive returns `degraded_mode.no_embeddings` or `degraded_mode.error`, the CLI exits 1 for one-shot asks. The console stays open so you can run `index` and retry.

---

## 6. Paper evaluation from the operator

`status` and console item `6` report NIST file presence plus last T1 run/score filenames. They do not score TREC.

```bash
./scripts/start-local.sh paper-eval
./scripts/start-local.sh paper-eval --stage score
./scripts/start-local.sh paper-eval --pipeline both --generator cloud --stage t3
```

Tenant is **`BENCHMARK_ORG_ID`** (default `bench-org`). `--org-id` is not applied to paper-eval. Full contract: [PAPER_EVAL.md](PAPER_EVAL.md).

---

## 7. Corpus and index

A compare is only honest when both pipelines read the same embedded ANN.

```bash
curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=LIT&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{"doc_id":"sample-lit-metformin","title":"Metformin renal guidance","text":"..."}]'

./scripts/start-local.sh index --org-id demo-org
curl -s 'http://127.0.0.1:8100/api/v1/storage/stats?org_id=demo-org'
curl -s http://127.0.0.1:8100/api/v1/naive/ready
```

You want `index_exists=true`, `embedded_count > 0`, and naive `retrieval_backend=ann` on a probe question. `mongo_cosine` is a local-dev fallback, not a published baseline. See [INDEXING.md](INDEXING.md).

---

## 8. Options

| Flag | Applies to | Default |
| --- | --- | --- |
| `--mode full\|naive\|both` | ask | `both` |
| `--question TEXT` | ask | REPL if omitted |
| `--org-id` / `--user-id` / `--patient-id` | ask, index | `demo-org` / `clinician-1` / empty |
| `--port` | all local API commands | `8100` |
| `--portal` | open | `clinician` |
| `--open` / `--no-open` | up | `up` opens browsers; `console` does not |
| `--json` | status / ask | human text |
| `--stage` / `--systems` / `--pipeline` / `--generator` / `--topic-field` | paper-eval | see [PAPER_EVAL.md](PAPER_EVAL.md) |
| `--stop-mongo` | stop | leave Compose Mongo running |

---

## 9. Troubleshooting

| Symptom | What to do |
| --- | --- |
| Console cannot import `app.cli.operator` | Run from the repository root; activate `.venv` |
| API never becomes healthy | `logs/medswin-api.log`; confirm Mongo and `.env` model URLs |
| Port already in use | `./scripts/start-local.sh status` then reuse, or `--port 8101` |
| Naive answers with “0 embeddings” | `index` or `POST /storage/embeddings/refresh` then rebuild |
| `eval` exits 1 | Use `paper-eval`. The `:8200` portal is gone |
| paper-eval NIST missing | `./scripts/start-local.sh warmup` |
| Full MedSwin degraded rerank | Missing reranker. Naive does not need one; a T3 pack does |
| Browsers did not open | `open clinician`, or paste the URLs from `status` |

---

## 10. Tests

```bash
python3 -m pytest tests/test_operator_cli.py tests/test_naive_rag.py tests/test_paper_eval_cli.py -q
```
