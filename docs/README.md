# MedSwin documentation

This folder is the governed manual set for the **live** runtime (`app/medswin/`, `app/cli/`, `scripts/start-local.sh`).

**Administrators start here:** [`ADMIN.md`](ADMIN.md) — configure `.env`, start, ingest, index, ask, compare, evaluate, and what not to follow.

Then open the specialist page that matches the job.

| Document | What it is for |
| --- | --- |
| [`ADMIN.md`](ADMIN.md) | Day-one runbook: config, ports, orgs, portals, first-run, eval, troubleshooting |
| [`OPERATOR.md`](OPERATOR.md) | Local console commands, menu, flags |
| [`NAIVE_RAG.md`](NAIVE_RAG.md) | Fairness contract for the textbook RAG control |
| [`ENDPOINTS.md`](ENDPOINTS.md) | HTTP routes for MedSwin, naive-RAG, storage, dashboard, eval |
| [`MEDSWIN.md`](MEDSWIN.md) | MAC, gate, scoring, traces (current runtime, not the old 3-agent sketch) |
| [`INDEXING.md`](INDEXING.md) | Embeddings, HNSW ∪ IVF, refresh / rebuild |
| [`../README.md`](../README.md) | Product overview, pipeline diagrams, data model |
| [`../eval/README.md`](../eval/README.md) | TREC CDS 2016 harness and MSAS |
| [`../env.example`](../env.example) | Environment keys. Copy to `.env`; never commit secrets |

The architecture paper (`MedSwin.tex`) is intentionally not the git-tracked runtime contract. Code in `app/` wins when a sentence disagrees with a diagram.

### Canonical vs leftover paths

| Follow | Do not treat as the local contract |
| --- | --- |
| `./scripts/start-local.sh` | `lab/` (HPC research modules) |
| `APP_PORT=8100`, DB `medswin` | Root `docker-compose.yml` / `Dockerfile` (legacy **8000** / `medical_rag_db`) |
| `demo-org` (ask) / `bench-org` (eval) | `aws/deploy.sh` (legacy EC2 compose deploy) |

### Default local ports

| Process | Port | Role |
| --- | --- | --- |
| MedSwin API | **8100** | Chat, ingest, traces, clinician UI, dashboard |
| Eval harness | **8200** | System audit UI and `POST /api/run` |
| Supervisor LLM | 8000 | OpenAI-compatible chat used by naive-RAG and the supervisor |
| Specialist LLMs | 8001–8003 | Optional agent endpoints; cloud mode collapses them |
| Reranker | 8004 | Required for a fair **full** MedSwin compare |
| Embeddings | 8005 | Optional HTTP embedder; local mode can use `ModelManager` |
