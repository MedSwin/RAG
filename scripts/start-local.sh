#!/bin/bash

# MedSwin local operator.
# A TTY session becomes the interactive console. Non-interactive use stays
# `serve`. Legacy --prompt / --question flags still work.
#
# Evaluation warmup is strict and idempotent by default: the requested MedSwin
# snapshot, the complete TREC-CDS 2016 cache, and Azure Foundry services are
# verified before normal startup. Set EVAL_WARMUP_ON_START=false or pass
# --skip-eval-warmup only when deliberately running a non-evaluation dev stack.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMAND=""
MODE="both"
QUESTION=""
PROMPT=0
ORG_ID="${ORG_ID:-demo-org}"
USER_ID="${USER_ID:-clinician-1}"
PATIENT_ID="${PATIENT_ID:-}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8100}"
EVAL_PORT="${EVAL_PORT:-8200}"
CLI_PORT=""
PORTAL="clinician"
PIPELINE="both"
CASES_PATH=""
MAX_CASES="2"
TOP_K=""
WITH_EVAL=0
OPEN_BROWSER=""
EVAL_ACTION="ui"
JSON=0
SKIP_EVAL_WARMUP=0
FORCE_EVAL_WARMUP=0
RESET_FULL_CORPUS=0

usage() {
    cat <<'EOF'
Usage:
  ./scripts/start-local.sh [command] [options]

Commands
  console     Interactive operator (default in a terminal)
  serve       API only, foreground (old default)
  up          Start API, print portals, open the clinician UI
  ask         Prompt full MedSwin, naive-RAG, or both
  eval        Start the benchmark UI, or run a compare
  warmup      Verify/download MedSwin model + full TREC-CDS + Foundry services
  full-eval   Prepare 100% TREC corpus/index and run the strict 2x2 eval matrix
  open        Open a portal in the browser
  status      API / corpus / naive / eval health
  index       Refresh embeddings and rebuild the ANN index
  stop        Stop API / eval processes started by this operator

Default
  Terminal (TTY)     → console: start API if needed, then a command menu
  Non-interactive    → serve:   uvicorn in the foreground

Options
  --mode MODE           full | naive | both          (ask)
  --question TEXT       One-shot question            (ask; implies ask)
  --prompt              Alias for ask (REPL if no --question)
  --org-id ID           Tenant org_id                (default: demo-org)
  --user-id ID          user_id                      (default: clinician-1)
  --patient-id ID       Optional EMR scope
  --port N              API port                     (default: 8100)
  --eval-port N         Eval portal port             (default: 8200)
  --pipeline P          medswin | naive_rag | both   (eval run)
  --cases-path PATH     Eval JSONL
  --max-cases N         Eval case cap
  --top-k N             Naive dense top-K
  --portal NAME         clinician|dashboard|eval|docs|all
  --with-eval           Also start the eval portal
  --open / --no-open    Open browsers
  --run                 With eval: run a benchmark instead of only the UI
  --skip-eval-warmup    Skip the three evaluation warmup tasks for this run
  --force-eval-warmup   Revalidate/redownload warmup artifacts
  --reset-full-corpus   Force full-eval to delete/rebuild the benchmark corpus
  --json                Machine-readable operator output
  --help

Evaluation defaults
  FOUNDRY_MODEL      gpt-5.4
  CLOUD_EMBEDDING    embed-v-4-0
  CLOUD_RERANKER     Cohere-rerank-v4.0-fast
  MedSwin generator  MedSwin/MedSwin-DaRE-TIES-KD-0.7

Examples
  ./scripts/start-local.sh
  ./scripts/start-local.sh warmup
  ./scripts/start-local.sh full-eval
  ./scripts/start-local.sh full-eval --reset-full-corpus
  ./scripts/start-local.sh up --with-eval --open
  ./scripts/start-local.sh ask --mode both --question "Can metformin continue?"
  ./scripts/start-local.sh eval --run --pipeline both --max-cases 2
  ./scripts/start-local.sh serve --skip-eval-warmup
EOF
}

KNOWN_COMMANDS="console serve up ask prompt eval warmup full-eval open status index stop"

if [[ $# -gt 0 && "$1" != -* ]]; then
    case " $KNOWN_COMMANDS " in
        *" $1 "*) COMMAND="$1"; shift ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT=1; shift ;;
        --question) QUESTION="${2:-}"; PROMPT=1; shift 2 ;;
        --mode) MODE="${2:-both}"; shift 2 ;;
        --org-id) ORG_ID="${2:-demo-org}"; shift 2 ;;
        --user-id) USER_ID="${2:-clinician-1}"; shift 2 ;;
        --patient-id) PATIENT_ID="${2:-}"; shift 2 ;;
        --port) CLI_PORT="${2:-8100}"; APP_PORT="$CLI_PORT"; shift 2 ;;
        --eval-port) EVAL_PORT="${2:-8200}"; shift 2 ;;
        --pipeline) PIPELINE="${2:-both}"; shift 2 ;;
        --cases-path) CASES_PATH="${2:-}"; shift 2 ;;
        --max-cases) MAX_CASES="${2:-2}"; shift 2 ;;
        --top-k) TOP_K="${2:-}"; shift 2 ;;
        --portal) PORTAL="${2:-clinician}"; shift 2 ;;
        --with-eval) WITH_EVAL=1; shift ;;
        --open) OPEN_BROWSER="1"; shift ;;
        --no-open) OPEN_BROWSER="0"; shift ;;
        --run) EVAL_ACTION="run"; shift ;;
        --skip-eval-warmup) SKIP_EVAL_WARMUP=1; shift ;;
        --force-eval-warmup) FORCE_EVAL_WARMUP=1; shift ;;
        --reset-full-corpus) RESET_FULL_CORPUS=1; shift ;;
        --json) JSON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

if [[ "$COMMAND" == "prompt" ]]; then
    COMMAND="ask"
    PROMPT=1
fi
if [[ "$PROMPT" -eq 1 && -z "$COMMAND" ]]; then
    COMMAND="ask"
fi
if [[ -z "$COMMAND" ]]; then
    if [[ -t 0 && -t 1 ]]; then
        COMMAND="console"
    else
        COMMAND="serve"
    fi
fi

if [[ "$MODE" != "full" && "$MODE" != "naive" && "$MODE" != "both" ]]; then
    echo -e "${RED}--mode must be full, naive, or both${NC}"
    exit 1
fi
if [[ "$PIPELINE" != "medswin" && "$PIPELINE" != "naive_rag" && "$PIPELINE" != "both" ]]; then
    echo -e "${RED}--pipeline must be medswin, naive_rag, or both${NC}"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Python 3 is not installed.${NC}"
    exit 1
fi

if [ -d ".venv" ]; then
    VENV=".venv"
elif [ -d "venv" ]; then
    VENV="venv"
else
    echo -e "${YELLOW}Creating virtual environment at .venv ...${NC}"
    python3 -m venv .venv
    VENV=".venv"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

if [ -n "$CLI_PORT" ]; then
    APP_PORT="$CLI_PORT"
fi
APP_PORT="${APP_PORT:-8100}"
EVAL_PORT="${EVAL_PORT:-8200}"
BASE_URL="${MEDSWIN_BASE_URL:-http://127.0.0.1:${APP_PORT}}"
EVAL_URL="${EVAL_BASE_URL:-http://127.0.0.1:${EVAL_PORT}}"

# Canonical evaluation model identities. Explicit .env values may override only
# where the caller deliberately wants a different deployment name/endpoint.
export FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5.4}"
export CLOUD_MODEL="${CLOUD_MODEL:-$FOUNDRY_MODEL}"
export CLOUD_EMBEDDING="${CLOUD_EMBEDDING:-embed-v-4-0}"
export CLOUD_EMBEDDING_DIMENSION="${CLOUD_EMBEDDING_DIMENSION:-1536}"
export CLOUD_RERANKER="${CLOUD_RERANKER:-Cohere-rerank-v4.0-fast}"
export CLOUD_MODEL_INFERENCE_API_VERSION="${CLOUD_MODEL_INFERENCE_API_VERSION:-2024-05-01-preview}"
export MEDSWIN_MODEL_REPO="${MEDSWIN_MODEL_REPO:-MedSwin/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_MODEL_PATH="${MEDSWIN_MODEL_PATH:-./models/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_LLM_MODEL="${MEDSWIN_LLM_MODEL:-MedSwin/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_LLM_URL="${MEDSWIN_LLM_URL:-http://127.0.0.1:8000/v1/chat/completions}"
export EVAL_WARMUP_ON_START="${EVAL_WARMUP_ON_START:-true}"

need_full_bootstrap() {
    case "$COMMAND" in
        stop|status|open) return 1 ;;
        *) return 0 ;;
    esac
}

run_eval_warmup() {
    local args=()
    if [[ "$FORCE_EVAL_WARMUP" -eq 1 ]]; then args+=(--force); fi
    echo -e "${YELLOW}Evaluation warmup: MedSwin snapshot + full TREC-CDS 2016 + Azure Foundry ...${NC}"
    CLOUD_MODE=true python3 scripts/warmup-eval.py "${args[@]}"
}

if need_full_bootstrap; then
    if ! python3 -c "import fastapi, uvicorn, httpx, pymongo, huggingface_hub" >/dev/null 2>&1; then
        echo -e "${YELLOW}Installing Python dependencies ...${NC}"
        pip install -r requirements.txt
    fi
    if ! python3 -c "import ir_datasets" >/dev/null 2>&1; then
        echo -e "${YELLOW}Installing TREC dataset adapter ...${NC}"
        pip install 'ir-datasets==0.5.9'
    fi

    export MONGODB_URL="${MONGODB_URL:-mongodb://localhost:27017}"
    export MONGODB_DB="${MONGODB_DB:-medswin}"
    export MONGODB_DATABASE="${MONGODB_DATABASE:-medswin}"

    echo -e "${YELLOW}Checking MongoDB ...${NC}"
    if ! python3 -c "import os, pymongo; pymongo.MongoClient(os.environ['MONGODB_URL']).admin.command('ping')" >/dev/null 2>&1; then
        echo -e "${YELLOW}MongoDB is not reachable. Starting mongo:6.0 with Docker ...${NC}"
        if ! command -v docker >/dev/null 2>&1; then
            echo -e "${RED}Docker is required to start MongoDB automatically.${NC}"
            exit 1
        fi
        if docker ps -a --format '{{.Names}}' | grep -q '^rag_mongodb$'; then
            docker start rag_mongodb >/dev/null
        else
            docker run -d -p 27017:27017 --name rag_mongodb mongo:6.0
        fi
        sleep 8
    fi

    mkdir -p models data logs storage
    if [ ! -d "models/MedEmbed-large-v0.1" ]; then
        echo -e "${YELLOW}Local embedding model not found at models/MedEmbed-large-v0.1${NC}"
        echo -e "${YELLOW}That is OK in CLOUD_MODE. Otherwise place the model or set EMBEDDING_URL.${NC}"
    fi
    if [ ! -d "models/bge-reranker-v2-m3" ]; then
        echo -e "${YELLOW}Local reranker not found at models/bge-reranker-v2-m3; full cloud evaluation uses Cohere rerank instead.${NC}"
    fi

    export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-./models/MedEmbed-large-v0.1}"
    export RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-./models/bge-reranker-v2-m3}"
    export HNSW_INDEX_PATH="${HNSW_INDEX_PATH:-./data/hnsw_index.bin}"
    export HNSW_MAPPING_PATH="${HNSW_MAPPING_PATH:-./data/hnsw_mapping.json}"
    export DATA_DIR="${DATA_DIR:-./data}"
    export DEBUG="${DEBUG:-true}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export APP_HOST
    export APP_PORT
    export EVAL_PORT

    if [[ "$COMMAND" != "warmup" && "$COMMAND" != "full-eval" && "$SKIP_EVAL_WARMUP" -eq 0 && "$EVAL_WARMUP_ON_START" == "true" ]]; then
        run_eval_warmup
    fi
fi

operator() {
    local extra=()
    extra+=(--base-url "$BASE_URL" --eval-url "$EVAL_URL")
    extra+=(--host "$APP_HOST" --port "$APP_PORT" --eval-port "$EVAL_PORT")
    extra+=(--mode "$MODE" --org-id "$ORG_ID" --user-id "$USER_ID")
    extra+=(--pipeline "$PIPELINE" --max-cases "$MAX_CASES" --portal "$PORTAL")
    extra+=(--eval-action "$EVAL_ACTION")
    if [ -n "$PATIENT_ID" ]; then extra+=(--patient-id "$PATIENT_ID"); fi
    if [ -n "$QUESTION" ]; then extra+=(--question "$QUESTION"); fi
    if [ -n "$TOP_K" ]; then extra+=(--top-k "$TOP_K"); fi
    if [ -n "$CASES_PATH" ]; then extra+=(--cases-path "$CASES_PATH"); fi
    if [ "$WITH_EVAL" -eq 1 ]; then extra+=(--with-eval); fi
    if [ "$OPEN_BROWSER" = "1" ]; then extra+=(--open); fi
    if [ "$OPEN_BROWSER" = "0" ]; then extra+=(--no-open); fi
    if [ "$JSON" -eq 1 ]; then extra+=(--json); fi
    python3 -m app.cli.operator "$@" "${extra[@]}"
}

print_banner() {
    echo -e "${GREEN}MedSwin local stack${NC}"
    echo -e "${GREEN}  API            ${BASE_URL}${NC}"
    echo -e "${GREEN}  Clinician UI   ${BASE_URL}/app/${NC}"
    echo -e "${GREEN}  Ops dashboard  ${BASE_URL}/api/v1/dashboard/${NC}"
    echo -e "${GREEN}  OpenAPI        ${BASE_URL}/docs${NC}"
    echo -e "${GREEN}  Eval portal    ${EVAL_URL}/${NC}"
    echo
    echo -e "${YELLOW}In another terminal:${NC}"
    echo -e "  ./scripts/start-local.sh ask --mode both"
    echo -e "  ./scripts/start-local.sh eval --with-eval --open"
    echo -e "  ./scripts/start-local.sh full-eval"
    echo -e "  ./scripts/start-local.sh status"
}

run_full_eval() {
    local bench_org="${BENCHMARK_ORG_ID:-bench-org}"
    local checkpoint="data/eval-warmup/full-trec-${bench_org}-checkpoint.json"
    local prep_args=(--org-id "$bench_org")
    run_eval_warmup
    # A first publication build must never mix an old smoke/sample corpus with
    # the complete corpus. Once a compatible checkpoint exists, resume/reuse it.
    if [[ "$RESET_FULL_CORPUS" -eq 1 || ! -f "$checkpoint" ]]; then
        prep_args+=(--reset)
    fi
    echo -e "${YELLOW}Preparing complete TREC-CDS runtime corpus and indexes ...${NC}"
    CLOUD_MODE=true CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE=query \
        python3 eval/scripts/prepare_full_trec_runtime.py "${prep_args[@]}"
    echo -e "${YELLOW}Running strict naive/full × MedSwin/GPT-5.4 matrix ...${NC}"
    CLOUD_MODE=true CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE=query \
        python3 eval/scripts/run_full_matrix.py --org-id "$bench_org"
}

case "$COMMAND" in
    serve)
        print_banner
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        python3 -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
        ;;
    console) operator console ;;
    up) operator up ;;
    ask) operator ask ;;
    eval) operator eval ;;
    warmup) run_eval_warmup ;;
    full-eval) run_full_eval ;;
    open) operator open ;;
    status) operator status ;;
    index) operator index ;;
    stop) operator stop ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac
