# MedSwin App

Production FastAPI runtime for MedSwin clinician decision support. The app implements enterprise medical RAG with calibrated retrieval, facet-level evidence sufficiency, structured multi-agent artifacts, PHI-safe audit traces, and bounded failure behavior. Methodological justification is covered in `docs/MedSwin.tex`.

## Runtime Architecture

- FastAPI service on `APP_PORT` with `/api/v1/*` routes.
- MongoDB stores documents, chunks, sessions, traces, policy decisions, and evidence provenance.
- External model services use OpenAI-compatible HTTP endpoints:
  - `SUPERVISOR_URL`
  - `AGENT1_URL`
  - `AGENT2_URL`
  - `AGENT3_URL`
  - `RERANKER_URL`
  - `EMBEDDING_URL`
- Retrieval uses dense ANN plus optional BM25, calibrated reranking, clinical fusion, budgeted evidence selection, and deterministic sufficiency gates.
- MedSwin responses are clinician decision support only and must not claim autonomous diagnosis.

## Core Guarantees

- `org_id` scopes sessions, traces, chunks, documents, and retrieval.
- `patient_id` scopes EMR evidence when provided.
- Evidence acceptance is facet-level: guideline concordance, safety/contraindications, patient applicability, and evidence quality are policy-gated.
- Raw source counts remain summary metrics only; they do not authorize generation.
- Final answers include citations, uncertainty, safety notes, and policy artifacts.
- Insufficient or conflicting evidence returns a bounded CDS response instead of unsupported recommendations.
- Trace summaries are PHI-redacted by default.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
docker run -d -p 27017:27017 --name mongodb mongo:6.0
python3 -m uvicorn app.main:app --reload --port 8100
```

API docs: `http://localhost:8100/docs`

Health: `GET /health`

## Main Endpoints

- `POST /api/v1/medswin/chat`
- `GET /api/v1/medswin/sessions/{session_id}`
- `GET /api/v1/medswin/traces/{trace_id}`
- `POST /api/v1/medswin/ingest`
- Legacy RAG endpoints remain under `/api/v1/preprocessing`, `/api/v1/embedding`, `/api/v1/retrieval`, and `/api/v1/storage`.

See `docs/ENDPOINTS.md` for request and response contracts.

## Key Configuration

| Variable | Purpose |
| --- | --- |
| `APP_PORT` | FastAPI port, default `8100` |
| `MONGODB_URL` / `MONGODB_DB` | MongoDB connection |
| `CANDIDATE_K` / `CANDIDATE_K_PRIME` | Retrieval candidate pool sizes |
| `MAX_RETRIEVE_LOOPS` | Maximum policy-driven retrieve-more loops |
| `TOKEN_BUDGET_B` | Evidence token budget |
| `SUFF_FACET_THRESHOLD` | Default facet coverage threshold |
| `SUFF_CRITICAL_FACET_THRESHOLD` | Critical facet coverage threshold |
| `SUFF_LCB_MARGIN` | Conservative calibration margin for coverage |
| `SUFF_MAX_ENTROPY` | Maximum allowed facet uncertainty |
| `SUFF_MAX_CONTRADICTIONS` | Accepted unresolved contradiction count |
| `W_RERANK`, `W_DENSE`, `W_LEX` | Retrieval scoring signals |
| `W_EBM`, `W_NOISE` | Evidence hierarchy and noise penalties |
| `TRACE_REDACT_BY_DEFAULT` | Redact trace summaries by default |

Invalid enterprise policy thresholds fail startup.

## Testing

```bash
python3 -m pytest
```

Use `python3` for all test commands.

## Important Files

- `app/models/medswin.py`: typed artifacts and public response models.
- `app/services/medswin/policy.py`: facet sufficiency, contradiction, and retrieve-more policy.
- `app/services/medswin/retrieval.py`: retrieval, calibrated fusion, and budgeted evidence selection.
- `app/services/medswin/orchestrator.py`: MAC workflow and bounded CDS answer path.
- `app/services/medswin/governance.py`: PHI redaction, citations, CDS boundary helpers.
- `docs/ENDPOINTS.md`: concise API contract.
