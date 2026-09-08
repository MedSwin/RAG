#!/usr/bin/env bash
# MedSwin 2.0 local operator and unified system-evaluation launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/start-local.sh [command] [options]

Commands
  console                 interactive local operator (TTY default)
  serve | up | ask         application service/operator commands
  system-eval              unified 3×2 benchmark
  warmup                   prepare evaluator/runtime dependencies
  open | status | index | stop

Application options
  --mode naive|two-stage|full|all       (default: all)
  --question TEXT --org-id ID --user-id ID --patient-id ID --port N

system-eval options
  --phase prepare|run|score|report|all  (default: all)
  --pipeline naive|two-stage|full|all   (default: all)
  --generator cloud|medswin|all         (default: all)
  --custom combination-september
  --smoke
  --replicates N --run-id ID

Examples
  ./scripts/start-local.sh system-eval
  ./scripts/start-local.sh system-eval --pipeline two-stage --generator medswin
  ./scripts/start-local.sh system-eval --custom combination-september
  ./scripts/start-local.sh system-eval --smoke
EOF
}

COMMAND=""
MODE="all"
QUESTION=""
ORG_ID="${ORG_ID:-demo-org}"
USER_ID="${USER_ID:-clinician-1}"
PATIENT_ID=""
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8100}"
PORTAL="clinician"
JSON=0
EVAL_PHASE="all"
EVAL_PIPELINE="all"
EVAL_GENERATOR="all"
EVAL_CUSTOM=""
EVAL_SMOKE=0
EVAL_REPLICATES=3
EVAL_RUN_ID=""

case "${1:-}" in
  paper-eval|full-eval|eval)
    echo "Removed command: $1"
    echo "Use: ./scripts/start-local.sh system-eval"
    echo "Example: ./scripts/start-local.sh system-eval --pipeline naive --generator cloud"
    exit 1
    ;;
  console|serve|up|ask|prompt|system-eval|warmup|open|status|index|stop)
    COMMAND="$1"; shift
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?missing mode}"; shift 2 ;;
    --question) QUESTION="${2:?missing question}"; COMMAND="ask"; shift 2 ;;
    --org-id) ORG_ID="${2:?missing org id}"; shift 2 ;;
    --user-id) USER_ID="${2:?missing user id}"; shift 2 ;;
    --patient-id) PATIENT_ID="${2:?missing patient id}"; shift 2 ;;
    --port) APP_PORT="${2:?missing port}"; shift 2 ;;
    --portal) PORTAL="${2:?missing portal}"; shift 2 ;;
    --json) JSON=1; shift ;;
    --phase) EVAL_PHASE="${2:?missing phase}"; shift 2 ;;
    --pipeline) EVAL_PIPELINE="${2:?missing pipeline}"; shift 2 ;;
    --generator) EVAL_GENERATOR="${2:?missing generator}"; shift 2 ;;
    --custom) EVAL_CUSTOM="${2:?missing custom preset}"; shift 2 ;;
    --smoke) EVAL_SMOKE=1; shift ;;
    --replicates) EVAL_REPLICATES="${2:?missing replicate count}"; shift 2 ;;
    --run-id) EVAL_RUN_ID="${2:?missing run id}"; shift 2 ;;
    --stage|--systems|--topic-field|--allow-local-t3|--reset-full-corpus)
      echo "Removed evaluation flag: $1"
      echo "Use: ./scripts/start-local.sh system-eval --phase all --pipeline all --generator all"
      exit 1
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  if [[ -t 0 && -t 1 ]]; then COMMAND="console"; else COMMAND="serve"; fi
fi
[[ "$COMMAND" == "prompt" ]] && COMMAND="ask"

if [[ "$MODE" != "naive" && "$MODE" != "two-stage" && "$MODE" != "full" && "$MODE" != "all" ]]; then
  echo "--mode must be naive, two-stage, full, or all"; exit 1
fi
if [[ -n "$EVAL_CUSTOM" && ( "$EVAL_PIPELINE" != "all" || "$EVAL_GENERATOR" != "all" ) ]]; then
  echo "--custom is mutually exclusive with --pipeline and --generator"; exit 1
fi

if [[ -d .venv ]]; then VENV=.venv; elif [[ -d venv ]]; then VENV=venv; else python3 -m venv .venv; VENV=.venv; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Load .env via Python so JSON-array values (ALLOWED_ORIGINS, ALLOWED_FILE_TYPES)
# are not broken by bash `source`.
if [[ -f .env ]]; then
  eval "$(python - <<'PY'
from pathlib import Path
import shlex
try:
    from dotenv import dotenv_values
    values = dotenv_values(".env")
except Exception:
    values = {}
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
for key, value in values.items():
    if value is None:
        continue
    print(f"export {shlex.quote(key)}={shlex.quote(value)}")
PY
)"
fi

export FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5.4}"
export CLOUD_MODEL="${CLOUD_MODEL:-$FOUNDRY_MODEL}"
export CLOUD_EMBEDDING="${CLOUD_EMBEDDING:-embed-v-4-0}"
export CLOUD_EMBEDDING_DIMENSION="${CLOUD_EMBEDDING_DIMENSION:-1536}"
export CLOUD_RERANKER="${CLOUD_RERANKER:-Cohere-rerank-v4.0-fast}"
export MEDSWIN_LLM_MODEL="${MEDSWIN_LLM_MODEL:-MedSwin/MedSwin-DaRE-TIES-KD-0.7}"
BASE_URL="${MEDSWIN_BASE_URL:-http://127.0.0.1:${APP_PORT}}"

operator() {
  local args=(--base-url "$BASE_URL" --host "$APP_HOST" --port "$APP_PORT" --mode "$MODE" --org-id "$ORG_ID" --user-id "$USER_ID" --portal "$PORTAL")
  [[ -n "$QUESTION" ]] && args+=(--question "$QUESTION")
  [[ -n "$PATIENT_ID" ]] && args+=(--patient-id "$PATIENT_ID")
  [[ "$JSON" -eq 1 ]] && args+=(--json)
  python -m app.cli.operator "$@" "${args[@]}"
}

case "$COMMAND" in
  serve) python -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" ;;
  console|up|ask|open|status|index|stop) operator "$COMMAND" ;;
  warmup) python scripts/warmup-eval.py --system-eval ;;
  system-eval)
    args=(--phase "$EVAL_PHASE" --pipeline "$EVAL_PIPELINE" --generator "$EVAL_GENERATOR" --replicates "$EVAL_REPLICATES")
    [[ -n "$EVAL_CUSTOM" ]] && args+=(--custom "$EVAL_CUSTOM")
    [[ "$EVAL_SMOKE" -eq 1 ]] && args+=(--smoke)
    [[ -n "$EVAL_RUN_ID" ]] && args+=(--run-id "$EVAL_RUN_ID")
    CLOUD_MODE=true SYSTEM_EVAL_MODE=true python -m benchmarks.system_eval "${args[@]}"
    ;;
esac
