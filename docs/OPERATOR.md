# Operator guide

Use `./scripts/start-local.sh` for the interactive console. Ask with `--mode naive|two-stage|full|all`; `all` compares all three responses through `/api/v1/pipelines/compare`. Status probes `/health`, `/api/v1/pipelines/ready`, storage, and the local MedSwin server. Benchmark execution is `./scripts/start-local.sh system-eval`; it is not an interactive review workflow.
