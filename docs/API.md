# API v2

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/api/v1/pipelines/{naive|two-stage|full}/chat` | Typed pipeline execution |
| POST | `/api/v1/pipelines/compare` | Generic comparison |
| GET | `/api/v1/pipelines/ready` | v2 profile readiness |
| GET | `/api/v1/pipelines/traces/{trace_id}` | Redacted trace |
| GET | `/api/v1/pipelines/sessions/{session_id}` | Session summary |

Responses include `pipeline`, `generator_profile`, `answer_artifact`, `ranked_evidence`, evidence, policy, degradations, timing, and trace ID. Benchmark metadata is accepted only with `SYSTEM_EVAL_MODE=true`. Ingestion uses `/api/v1/storage/chunks`.
