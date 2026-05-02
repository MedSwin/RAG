# MedSwin Endpoints

Base URL: `/api/v1`

All MedSwin endpoints require `org_id` for tenant scoping. EMR retrieval is additionally scoped by `patient_id` when provided.

## `POST /medswin/chat`

Runs the clinician decision-support pipeline.

Request:

```json
{
  "query": "What treatment options are safe for this patient?",
  "user_id": "user-1",
  "org_id": "org-1",
  "session_id": "optional-session-id",
  "patient_id": "optional-patient-id",
  "constraints": {
    "clinical_scope": "clinician_cds",
    "guideline_only": false,
    "required_facets": [],
    "source_policy": "ANY",
    "min_evidence_grade": 0.7,
    "timeframe": "optional",
    "specialties": ["cardiology"]
  }
}
```

Response includes:

- `answer`: clinician CDS response; never final diagnosis.
- `evidence_bundle`: selected passages, token count, source counts, facet coverage, evidence ledger, contradictions, and policy decision.
- `policy_decision`: top-level generation gate.
- `facet_coverage`: calibrated facet coverage with LCB and entropy.
- `contradictions`: unresolved evidence conflicts.
- `evidence_ledger`: claim-level provenance.
- `citations`: chunk/document/source/section/version metadata.
- `degraded_mode`: service failure flags.

If evidence is insufficient, the endpoint returns `200` with a bounded insufficient-evidence CDS answer and `policy_decision.passed=false`.

## `GET /medswin/sessions/{session_id}`

Returns a scoped session summary.

Query:

- `org_id`: required.

Response:

```json
{
  "session_id": "session-id",
  "user_id": "user-id",
  "org_id": "org-id",
  "created_at": "2026-05-02T00:00:00",
  "last_active": "2026-05-02T00:00:00",
  "metadata": {}
}
```

## `GET /medswin/traces/{trace_id}`

Returns a PHI-safe trace summary.

Query:

- `org_id`: required.
- `include_details`: optional; includes policy details only when enabled by server config.

Response includes trace/session IDs, redacted query, message/tool counts, sufficiency-check count, evidence count, and optional policy artifacts.

## `POST /medswin/ingest`

Ingests CPG, EMR, or literature documents.

Query:

- `source_type`: `CPG`, `EMR`, or `LIT`.
- `org_id`: required.

Body:

```json
[
  {
    "doc_id": "guideline-1",
    "title": "Guideline title",
    "version": "2026.1",
    "effective_date": "2026-01-01T00:00:00",
    "patient_id": "optional-for-emr",
    "source_reliability": 0.95,
    "evidence_grade": {
      "label": "guideline",
      "score": 0.95,
      "source_reliability": 0.95
    },
    "tags": ["diabetes"],
    "metadata": {},
    "text": "Recommendations\n\n...",
    "chunks": [
      {
        "chunk_id": "optional",
        "text": "Chunk text",
        "section": "Recommendations",
        "offset_start": 0,
        "offset_end": 120,
        "metadata": {}
      }
    ]
  }
]
```

If `chunks` is omitted, the app uses section-aware chunking and preserves headings, offsets, version, effective date, patient ID, evidence grade, source reliability, and tokenized text.

## Legacy RAG Endpoints

- `POST /preprocessing/chunk`
- `POST /preprocessing/upload-and-chunk`
- `GET /preprocessing/info`
- `POST /embedding/embed`
- `POST /embedding/embed/batch`
- `GET /embedding/info`
- `POST /retrieval/search`
- `GET /retrieval/search`
- `GET /retrieval/index/info`
- `POST /storage/chunks`
- `POST /storage/index/build`
- `GET /storage/stats`
- `DELETE /storage/chunks`

Legacy endpoints remain available for compatibility but do not enforce MedSwin enterprise sufficiency policy unless routed through `/medswin/chat`.
