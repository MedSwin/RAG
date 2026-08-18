# Local operator

This is the day-to-day manual for running MedSwin on one machine: start the API, ask the full system or the naive-RAG control, watch the same turn on the web portals, rebuild the index, and run the eval harness.

The entry point is [`scripts/start-local.sh`](../scripts/start-local.sh). The intelligence lives in [`app/cli/operator.py`](../app/cli/operator.py). Asking a question reuses [`app/cli/prompt.py`](../app/cli/prompt.py).

Day-one runbook (ports, `.env`, first ingest): [ADMIN.md](ADMIN.md).  
Related: [NAIVE_RAG.md](NAIVE_RAG.md), [eval/README.md](../eval/README.md), [ENDPOINTS.md](ENDPOINTS.md).

---

## 1. Surfaces

After the stack is up you should be able to reach all four of these. They share the same Mongo corpus and the same API process.

| Surface | URL | What you do there |
| --- | --- | --- |
| Clinician CDS | http://127.0.0.1:8100/app/ | Ask **full MedSwin**, **naive-RAG**, or **both**. Inspect answer, passages, facets, contradictions, traces |
| Ops dashboard | http://127.0.0.1:8100/api/v1/dashboard/ | Corpus / Hugging Face ingest / model download |
| OpenAPI | http://127.0.0.1:8100/docs | Live request schemas |
| Eval harness | http://127.0.0.1:8200/ | Batch audit of the same cases on one or both pipelines |
| Health | http://127.0.0.1:8100/health | Process liveness and loaded models |
| Naive preflight | http://127.0.0.1:8100/api/v1/naive/ready | Mongo, chunk count, embedding count, index file |

The clinician page served at `/app/` is the zero-build UI in [`web/public/index.html`](../web/public/index.html). If you run `npm run build` in `web/`, FastAPI prefers `web/dist/` instead.

Form fields: pipeline (`full` / `naive` / `both`), question, `patient_id`, `org_id` (default `demo-org`), `user_id` (default `clinician-1`), naive `top_k` (1–20). The UI does **not** send `session_id` or `constraints` — use curl or eval for those. Nav links: clinician, ops dashboard, OpenAPI, eval portal (`hostname:8200`).

---

## 2. Commands

```text
./scripts/start-local.sh [command] [options]
```

| Command | Behaviour |
| --- | --- |
| `console` | Default in a TTY. Ensures venv / Mongo / API, prints portal URLs, opens a command menu |
| `serve` | Default when stdin is not a TTY. `uvicorn app.main:app` in the foreground |
| `up` | Start the API (and optionally eval), print status, open the clinician and dashboard browsers |
| `ask` | One question or a REPL. Modes: `full`, `naive`, `both` |
| `eval` | Start the :8200 portal, or `--run` a benchmark against the live API |
| `open` | Open `clinician`, `dashboard`, `eval`, `docs`, or `all` |
| `status` | API health, naive ready, storage stats, eval portal |
| `index` | `POST /storage/embeddings/refresh` then `POST /storage/index/build` |
| `stop` | SIGTERM the API / eval PIDs this operator wrote under `logs/` |

Legacy flags still work:

```bash
./scripts/start-local.sh --prompt
./scripts/start-local.sh --mode both --question "Can metformin continue?"
```

Those are aliases for `ask`.

### Defaults

- TTY session → `console`
- Piped / CI session → `serve`
- API port `8100`, eval port `8200`, org `demo-org`, user `clinician-1`
- `--port` wins over `APP_PORT` from `.env`

### Typical sessions

```bash
# Daily local work
./scripts/start-local.sh
# then type a question, or: 3   (compare both)
# then: open clinician

# Demo: browsers open, eval portal included
./scripts/start-local.sh up --with-eval --open

# One-shot compare without the menu
./scripts/start-local.sh ask --mode both --question "What first-line therapy is appropriate?" --patient-id patient-7

# Smoke eval (needs gold docs in BENCHMARK_ORG_ID)
./scripts/start-local.sh eval --run --pipeline both --max-cases 2

# Rebuild vectors + ANN after ingest
./scripts/start-local.sh index --org-id demo-org

# Leave the web UIs up; later:
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
  6  open eval         starts :8200 if needed
  7  eval run          eval.app.runner against the live API
  8  index             refresh embeddings + rebuild ANN
  9  status
  h  help
  q  quit              leaves servers running
```

You can also type a clinical question (three or more words, or a trailing `?`) and the console uses the current default mode (`mode full|naive|both` changes it).

`q` does **not** kill uvicorn. The clinician UI and eval portal stay reachable. Use `stop` when you want them down. PIDs live in:

```text
logs/medswin-api.pid
logs/medswin-api.log
logs/medswin-eval.pid
logs/medswin-eval.log
```

Only processes this operator started are stopped. An API you launched yourself with `serve` is left alone.

---

## 4. What the bootstrap does

`start-local.sh` (for commands other than `status` / `open` / `stop`):

1. Finds `.venv` or `venv`, or creates `.venv`
2. Loads `.env` if present
3. Installs `requirements.txt` only if `fastapi`, `uvicorn`, `httpx`, or `pymongo` are missing
4. Pings `MONGODB_URL` (default `mongodb://localhost:27017`). If down, starts `mongo:6.0` as `rag_mongodb`
5. Forces `MONGODB_DB=medswin` unless you already exported another name
6. Exports `DEBUG=true` unless you already set `DEBUG` in the environment (overrides `env.example` `DEBUG=false`)
7. Creates `models/`, `data/`, `logs/`, `storage/`
8. Warns if `models/MedEmbed-large-v0.1` or `models/bge-reranker-v2-m3` are absent
9. Hands off to `python -m app.cli.operator …` or, for `serve`, to uvicorn

`EVAL_BASE_URL` (default `http://127.0.0.1:8200`) is read by the shell, not by `app/core/config.py`.

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

The clinician UI uses the same three routes. A compare turn shows naive and MedSwin side by side plus the Jaccard / abstention diff.

Shared request fields: `query`, `user_id`, `org_id`, optional `patient_id`, `session_id`, `constraints`. Naive also accepts `top_k` (default `NAIVE_TOP_K=5`).

Patient notes belong in EMR ingest, not pasted into the question, unless you deliberately turn on eval's `include_patient_context_in_query`.

If naive returns `degraded_mode.no_embeddings` or `degraded_mode.error`, the CLI exits 1 for one-shot asks. The console stays open so you can run `index` and retry.

---

## 6. Evaluation from the operator

```bash
./scripts/start-local.sh eval --open
./scripts/start-local.sh eval --run --pipeline both --max-cases 2 \
  --cases-path eval/data/sample/cases.jsonl
```

`eval` without `--run` starts `eval.app.main:app` on `:8200` and opens the benchmark UI.

`eval --run` calls the same runner as `POST http://127.0.0.1:8200/api/run`:

- `pipeline=medswin|naive_rag|both`
- tenant is **`BENCHMARK_ORG_ID`** (default `bench-org`). `--org-id` is **not** applied to eval runs
- naive preflight via `GET /api/v1/naive/ready`
- writes `{run_id}.json` and, for `both`, `{run_id}.comparison.json` under `RUN_STORE_DIR` (default `/tmp/medswin-audits`)

Smoke cases in `eval/data/sample/cases.jsonl` are **not** publication numbers. They also fail qrel coverage unless those gold `doc_id`s exist in `bench-org`. For TREC, follow [`eval/README.md`](../eval/README.md).

---

## 7. Corpus and index

A compare is only honest when both pipelines read the same embedded ANN.

```bash
# ingest (attaches embeddings when the embedder is reachable)
curl -s 'http://127.0.0.1:8100/api/v1/medswin/ingest?source_type=LIT&org_id=demo-org' \
  -H 'Content-Type: application/json' \
  -d '[{"doc_id":"sample-lit-metformin","title":"Metformin renal guidance","text":"..."}]'

# or from the console
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
| `--org-id` / `--user-id` / `--patient-id` | ask, index, eval | `demo-org` / `clinician-1` / empty |
| `--port` / `--eval-port` | all | `8100` / `8200` |
| `--pipeline medswin\|naive_rag\|both` | eval --run | `both` |
| `--cases-path` / `--max-cases` | eval --run | `eval/data/sample/cases.jsonl` / `2` |
| `--top-k` | naive ask / eval | runtime `NAIVE_TOP_K` |
| `--portal` | open | `clinician` |
| `--with-eval` | up | off |
| `--open` / `--no-open` | up, eval ui | `up` opens browsers; `console` does not |
| `--run` | eval | UI only unless set |
| `--json` | status / ask / eval | human text |

---

## 9. Troubleshooting

| Symptom | What to do |
| --- | --- |
| Console cannot import `app.cli.operator` | Run from the repository root; activate `.venv` |
| API never becomes healthy | `logs/medswin-api.log`; confirm Mongo and `.env` model URLs |
| Port already in use | `./scripts/start-local.sh status` then reuse, or `--port 8101` |
| Naive answers with “0 embeddings” | `index` or `POST /storage/embeddings/refresh` then rebuild |
| Eval portal 404 / down | `eval --open` from repo root (`python -m uvicorn eval.app.main:app --port 8200`) |
| Eval qrel coverage error | Smoke gold IDs are not in `bench-org`; ingest them or use TREC prep |
| Full MedSwin degraded rerank | Missing reranker. Naive does not need one; a paper compare does |
| Browsers did not open | `open clinician` / `open eval`, or paste the URLs from `status` |

---

## 10. Tests

```bash
python3 -m pytest tests/test_operator_cli.py tests/test_naive_rag.py tests/test_eval_harness.py -q
```
