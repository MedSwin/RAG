# MedSwin End-to-End System Benchmark

This harness evaluates the full MedSwin system rather than the standalone LLM or reranker. It calls an existing MedSwin FastAPI runtime, runs benchmark cases through `/api/v1/medswin/chat`, retrieves trace summaries, and emits an audit JSON file containing evidence, provenance, sufficiency, safety, and trace-completeness metrics.

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
        | ingest case note          -> POST /api/v1/medswin/ingest?source_type=EMR
        | chat call                 -> POST /api/v1/medswin/chat
        | trace call                -> GET  /api/v1/medswin/traces/{trace_id}
        v
MedSwin runtime :8100 (per-run org namespace)
        |
        v
Audit JSON: MSAS, facet recall, critical-facet recall, citation precision,
            sufficiency decision score, trace completeness, penalties
```

## File layout

```text
app/
  main.py              # FastAPI benchmark API + static UI
  runner.py            # end-to-end benchmark runner
  client.py            # calls MedSwin endpoints
  audit.py             # metrics and MSAS computation
  schemas.py           # Pydantic audit/case models
  config.py            # env-driven settings
scripts/
  prepare_trec_cds.py  # exports TREC CDS cases with qrels via ir_datasets
  ingest_trec_pmc.py   # bulk-ingests PMC evidence into MedSwin
data/sample/
  cases.jsonl          # two toy cases for smoke testing
audits/
  audit_schema.json    # expected audit output shape
static/
  index.html           # simple UI
Dockerfile
README.md
```

## Quick start

Start MedSwin first:

```bash
python3 -m uvicorn app.main:app --reload --port 8100
```

Then run this benchmark service:

```bash
cd eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m uvicorn app.main:app --reload --port 8200
```

Open:

```text
http://localhost:8200
```

Or run through Docker:

```bash
docker compose up --build
```

## Preparing TREC CDS 2016 cases

```bash
python scripts/prepare_trec_cds.py \
  --dataset pmc/v2/trec-cds-2016 \
  --out data/trec_cds_2016/cases.jsonl \
  --max-topics 30
```

The script seeds `gold_facets` from TREC qrels. For final paper numbers, refine the facet labels manually or with clinician adjudication, because qrels are document-level relevance judgments rather than facet-level clinical labels.

## Ingesting the PMC evidence corpus

For a smoke test:

```bash
python scripts/ingest_trec_pmc.py \
  --dataset pmc/v2 \
  --limit 1000 \
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
    "clinical_scope": "clinician_cds"
  }'
```

The output is saved to `audits/{run_id}.json`.

Each run uses a unique benchmark org derived from the configured `BENCHMARK_ORG_ID`, which prevents repeated evaluations from colliding on document IDs or patient-scoped retrieval state.

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

For publication, replace `groundedness_proxy` and `clinical_quality_proxy` with blinded clinician or rubric-based claim adjudication. Keep the automatic metrics as reproducible system diagnostics.

## Recommended paper comparison

Evaluate the same cases under these configurations:

1. LLM-only best model, no retrieval.
2. Naive RAG top-K.
3. RAG + MedSwin-Rerank.
4. Full MedSwin without MAC.
5. Full MedSwin.

This isolates whether the whole system improves evidence coverage, safety, provenance, and sufficiency behavior beyond model-level generation and reranking.
