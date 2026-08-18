# MedSwin End-to-End System Benchmark

Administrator first-run (API + operator + portals): [`docs/ADMIN.md`](../docs/ADMIN.md).

This harness evaluates the live MedSwin runtime rather than a standalone LLM or reranker. It runs the same cases through `/api/v1/medswin/chat` and/or the naive-RAG control at `/api/v1/naive/chat`, retrieves trace summaries, and emits an audit JSON file containing evidence, provenance, sufficiency, safety, and trace-completeness metrics.

The benchmark now keeps the clinical question text separate from the patient note. Case context is ingested as EMR-like evidence, while the query sent to MedSwin stays pure so retrieval quality is measured against the runtime contract rather than prompt augmentation.

## Recommended benchmark dataset

Use **TREC Clinical Decision Support 2016** as the primary publication benchmark.

Why this dataset:

- It is designed for clinical decision-support retrieval, not generic answer matching.
- The 2016 track uses real MIMIC-III admission notes rather than synthetic cases.
- It provides patient-context topics, generic clinical information needs, a large PubMed Central evidence corpus, and relevance judgments.
- It directly matches MedSwin's system claims: patient-specific context, biomedical evidence retrieval, provenance, and sufficiency gating.

MedQuAD and HealthBench remain useful for answer-generation benchmarking, but they are weaker for the missing whole-system audit because they do not provide the same patient-note + evidence-corpus + qrel structure.

## Architecture

```text
TREC CDS case JSONL
        |
        v
Benchmark FastAPI service :8200
        |
        | preflight health check    -> GET  /health
        | naive preflight (control) -> GET  /api/v1/naive/ready
        | ingest case note          -> POST /api/v1/medswin/ingest?source_type=EMR
        | chat call                 -> POST /api/v1/medswin/chat
        |                           or POST /api/v1/naive/chat  (same org, query, constraints)
        | trace call                -> GET  /api/v1/medswin/traces/{trace_id}
        v
MedSwin runtime :8100 (fixed benchmark org namespace)
        |
        v
Audit JSON: MSAS, facet recall, critical-facet recall, citation precision,
            sufficiency decision score, trace completeness, penalties
```

## File layout

Paths below are relative to **`eval/`**. From the repository root, prefix with `eval/`.

```text
eval/
  app/
    main.py              # FastAPI benchmark API + static UI
    runner.py            # end-to-end benchmark runner
    client.py            # calls MedSwin / naive endpoints
    audit.py             # metrics and MSAS computation
    schemas.py           # Pydantic audit/case models (default cases_path is repo-root relative)
    config.py            # env-driven settings
  scripts/
    prepare_trec_cds.py  # exports TREC CDS cases with qrels via ir_datasets
    ingest_trec_pmc.py   # bulk-ingests PMC evidence into MedSwin
    run_pipeline_compare.py
  data/sample/
    cases.jsonl          # two toy cases — from repo root: eval/data/sample/cases.jsonl
  audits/
    audit_schema.json
  static/
    index.html
  Dockerfile
  docker-compose.yml     # eval service only (:8200); MedSwin must already be on :8100
  README.md
```

## Quick start

From the repository root, the operator can start both portals:

```bash
./scripts/start-local.sh up --with-eval --open
# clinician UI  http://127.0.0.1:8100/app/
# eval portal   http://127.0.0.1:8200/
./scripts/start-local.sh eval --run --pipeline both --max-cases 2
```

Or start the services yourself **from the repository root**:

```bash
python3 -m uvicorn app.main:app --reload --port 8100
python3 -m uvicorn eval.app.main:app --reload --port 8200
```

`docker compose up --build` inside `eval/` starts **only** the harness on :8200. It expects MedSwin already healthy on the host at :8100. It is not a substitute for `./scripts/start-local.sh`.

Open:

```text
http://localhost:8200
```

Or run through Docker:

```bash
docker compose up --build
```

## Preparing TREC CDS 2016 cases

From `eval/` (or prefix with `eval/` from the repository root):

```bash
python scripts/prepare_trec_cds.py \
  --dataset pmc/v2/trec-cds-2016 \
  --out data/trec_cds_2016/cases.jsonl \
  --max-topics 30 \
  --max-docs-per-topic 200
```

The script seeds `gold_facets` from TREC qrels. For final paper numbers, refine the facet labels manually or with clinician adjudication, because qrels are document-level relevance judgments rather than facet-level clinical labels. Evidence documents are capped deterministically per topic, prioritizing higher relevance grades first and using `doc_id` as the stable tiebreaker.

## Ingesting the PMC evidence corpus

For a smoke test:

```bash
python scripts/ingest_trec_pmc.py \
  --dataset pmc/v2 \
  --limit 1000 \
  --medswin-base-url http://localhost:8100 \
  --org-id bench-org
```

For the default benchmark subset:

```bash
python3 scripts/ingest_trec_pmc.py \
  --dataset pmc/v2 \
  --sample-size 5000 \
  --seed 1337 \
  --reset-org \
  --build-index \
  --medswin-base-url http://localhost:8100 \
  --org-id bench-org
```

For final experiments, use the full TREC CDS evidence corpus or an explicitly documented judged-pool + hard-negative subset. Report the corpus construction in the paper.

## Running an audit

```bash
curl -X POST http://localhost:8200/api/run \
  -H 'Content-Type: application/json' \
  -d '{
    "cases_path": "data/trec_cds_2016/cases.jsonl",
    "max_cases": 30,
    "ingest_case_context": true,
    "source_policy": "ANY",
    "min_evidence_grade": 0.3,
    "clinical_scope": "clinician_cds",
    "pipeline": "medswin"
  }'
```

The output is saved under `RUN_STORE_DIR` (default `/tmp/medswin-audits/{run_id}.json`). `pipeline=both` also writes `{run_id}.comparison.json`.

Each run uses the fixed configured `BENCHMARK_ORG_ID` (default `bench-org`), so the prepared corpus remains visible to `/api/v1/medswin/chat` and `/api/v1/naive/chat`. Use `POST /api/v1/storage/benchmark/reset` or the ingest script's `--reset-org` option before preparing a fresh corpus.

### Eval HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Benchmark UI |
| `GET` | `/health` | Harness liveness |
| `GET` | `/api/cases?path=…` | Preview cases JSONL |
| `POST` | `/api/run` | Run an audit |
| `GET` | `/api/runs` | List audits (`pipeline`, MSAS, optional naive MSAS) |
| `GET` | `/api/runs/{run_id}` | Full `RunAudit` |
| `GET` | `/api/runs/{run_id}/download` | Download JSON |

`RunRequest` fields: `cases_path`, `max_cases`, `max_concurrency`, `reranker_budget`, `ingest_case_context`, `fetch_trace_summary`, `include_patient_context_in_query`, `source_policy`, `guideline_only`, `min_evidence_grade`, `clinical_scope`, `pipeline` (`medswin` \| `naive_rag` \| `both`), `top_k` (default 5, sent to `/naive/chat`).

From the repo root, start the UI with `./scripts/start-local.sh eval --open` so the default cases path `eval/data/sample/cases.jsonl` resolves.

## Main metric: MedSwin System Audit Score

The harness computes:

```text
MSAS = 0.25 critical_facet_recall
     + 0.15 facet_recall
     + 0.15 groundedness_proxy
     + 0.15 sufficiency_decision_score
     + 0.10 trace_completeness
     + 0.10 evidence_doc_recall
     + 0.10 clinical_quality_proxy
     - 0.20 unsafe_omission_penalty
     - 0.10 unsupported_penalty
```

> For publication, replace `groundedness_proxy` and `clinical_quality_proxy` with blinded clinician or rubric-based claim adjudication. Keep the automatic metrics as reproducible system diagnostics.

## Comparison

The runtime now exposes a first-class naive-RAG control (`POST /api/v1/naive/chat`) that reuses the same embeddings, ANN index, Mongo corpus, and LLM as `/medswin/chat`. Operator manual: [`docs/NAIVE_RAG.md`](../docs/NAIVE_RAG.md).

Evaluate the same cases under these configurations:

1. LLM-only best model, no retrieval (not implemented in this harness).
2. Naive RAG top-K — **implemented**: set `"pipeline": "naive_rag"` on `POST /api/run`.
3. RAG + MedSwin-Rerank (not a separate route; disable MAC/gate only by changing runtime code).
4. Full MedSwin without MAC (not a separate route).
5. Full MedSwin — **implemented**: `"pipeline": "medswin"` (default).

To run 2 and 5 back-to-back and write a delta file:

```bash
curl -X POST http://localhost:8200/api/run \
  -H 'Content-Type: application/json' \
  -d '{
    "cases_path": "eval/data/sample/cases.jsonl",
    "max_cases": 2,
    "pipeline": "both",
    "top_k": 5
  }'
```

From the repo root the smoke file is `eval/data/sample/cases.jsonl`. Smoke gold `doc_id`s must exist in `bench-org` or qrel coverage validation fails — that is intentional.

Or with MedSwin already healthy:

```bash
python3 eval/scripts/run_pipeline_compare.py \
  --cases-path eval/data/sample/cases.jsonl \
  --max-cases 2
```

This isolates whether the whole system improves evidence coverage, safety, provenance, and sufficiency behavior beyond model-level generation and naive top-K stuffing.

Fairness contract for the two pipelines:

- Same `BENCHMARK_ORG_ID`, cases file, EMR ingest, index provenance, qrel gates, query text, and `min_evidence_grade`.
- Naive chat sends `top_k` (default 5). MedSwin is not capped by that field.
- Naive concurrency follows `max_concurrency`. MedSwin case fan-out is still capped by `reranker_budget`.
- Naive infrastructure failures (`no_embeddings`, dim mismatch, runtime error) count toward `error_rate`. Missing MAC / gate artifacts do not.
- `pipeline=both` returns the MedSwin `RunAudit`. Naive totals and deltas are in `diagnostics.pipeline_comparison` and `{run_id}.comparison.json`. Do not publish a run whose naive `retrieval_backend_counts` are not `ann`.
