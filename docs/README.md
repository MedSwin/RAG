# MedSwin documentation

This folder is the operator and architecture manual set for the live runtime. Start here, then open the page that matches the job.

| Document | What it is for |
| --- | --- |
| [`../README.md`](../README.md) | Product overview, pipeline diagrams, data model, local setup |
| [`OPERATOR.md`](OPERATOR.md) | Local console: start, ask, compare, eval, portals, index, stop |
| [`NAIVE_RAG.md`](NAIVE_RAG.md) | Fairness contract and how to run the textbook RAG control |
| [`ENDPOINTS.md`](ENDPOINTS.md) | HTTP API contract for MedSwin, naive-RAG, storage, and ops |
| [`MEDSWIN.md`](MEDSWIN.md) | Current runtime architecture (MAC, gate, scoring, traces) |
| [`INDEXING.md`](INDEXING.md) | ANN files, ingest embeddings, BM25, and index rebuild |
| [`../eval/README.md`](../eval/README.md) | TREC CDS harness, MSAS, `pipeline=medswin\|naive_rag\|both` |
| [`../env.example`](../env.example) | Environment keys. Copy to `.env`. Never commit secrets |

The architecture paper (`MedSwin.tex`) is intentionally not tracked in this repository. The runtime in `app/medswin/` is the executable source of truth.

Default local ports:

| Process | Port | Role |
| --- | --- | --- |
| MedSwin API | **8100** | Chat, ingest, traces, clinician UI, dashboard |
| Eval harness | **8200** | System audit UI and `POST /api/run` |
| Supervisor LLM | 8000 | OpenAI-compatible chat used by naive-RAG and the supervisor |
| Specialist LLMs | 8001–8003 | Optional agent endpoints; cloud mode collapses them |
| Reranker | 8004 | Required for a fair **full** MedSwin compare |
| Embeddings | 8005 | Optional HTTP embedder; local mode can use `ModelManager` |
