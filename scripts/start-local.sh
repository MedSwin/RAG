#!/bin/bash

# Local MedSwin + naive-RAG startup and terminal prompt.
# Run from the repository root:
#   ./scripts/start-local.sh
#   ./scripts/start-local.sh --prompt
#   ./scripts/start-local.sh --question "Can metformin continue?" --mode both

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="both"
QUESTION=""
PROMPT=0
ORG_ID="${ORG_ID:-demo-org}"
USER_ID="${USER_ID:-clinician-1}"
PATIENT_ID="${PATIENT_ID:-}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8100}"
CLI_PORT=""
STARTED_SERVER=0
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage:
  ./scripts/start-local.sh [options]

Default (no flags): prepare the local environment and start the API in the
foreground on APP_PORT (default 8100).

Options:
  --prompt              Start the API if needed, then open an interactive REPL
  --question TEXT       One-shot question (implies --prompt)
  --mode MODE           full | naive | both   (default: both)
  --org-id ID           Tenant org_id         (default: demo-org)
  --user-id ID          user_id               (default: clinician-1)
  --patient-id ID       Optional patient scope
  --port N              API port              (default: 8100)
  --help                Show this help

Examples:
  ./scripts/start-local.sh
  ./scripts/start-local.sh --prompt
  ./scripts/start-local.sh --mode naive --question "What is first-line therapy?"
  ./scripts/start-local.sh --mode both --question "Can this patient continue metformin?" --patient-id patient-42

See docs/NAIVE_RAG.md for the full reproducibility checklist.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT=1; shift ;;
        --question) QUESTION="${2:-}"; PROMPT=1; shift 2 ;;
        --mode) MODE="${2:-both}"; shift 2 ;;
        --org-id) ORG_ID="${2:-demo-org}"; shift 2 ;;
        --user-id) USER_ID="${2:-clinician-1}"; shift 2 ;;
        --patient-id) PATIENT_ID="${2:-}"; shift 2 ;;
        --port) CLI_PORT="${2:-8100}"; APP_PORT="$CLI_PORT"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "full" && "$MODE" != "naive" && "$MODE" != "both" ]]; then
    echo -e "${RED}--mode must be full, naive, or both${NC}"
    exit 1
fi

echo -e "${GREEN}Starting Medical RAG local environment${NC}"

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

echo -e "${YELLOW}Activating ${VENV} ...${NC}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [ -f ".env" ]; then
    echo -e "${YELLOW}Loading .env ...${NC}"
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

if [ -n "$CLI_PORT" ]; then
    APP_PORT="$CLI_PORT"
fi
APP_PORT="${APP_PORT:-8100}"
BASE_URL="${MEDSWIN_BASE_URL:-http://127.0.0.1:${APP_PORT}}"

echo -e "${YELLOW}Installing Python dependencies ...${NC}"
pip install -r requirements.txt

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
    echo -e "${YELLOW}Local reranker not found at models/bge-reranker-v2-m3 (optional for naive-RAG).${NC}"
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

api_healthy() {
    python3 - "$BASE_URL" <<'PY'
import sys, urllib.request
base = sys.argv[1].rstrip("/")
try:
    urllib.request.urlopen(base + "/health", timeout=2)
except Exception:
    raise SystemExit(1)
PY
}

start_server_background() {
    echo -e "${GREEN}Starting API at ${BASE_URL} ...${NC}"
    python3 -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload &
    SERVER_PID=$!
    STARTED_SERVER=1
    for _ in $(seq 1 60); do
        if api_healthy; then
            echo -e "${GREEN}API is healthy.${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}API did not become healthy within 60s.${NC}"
    return 1
}

stop_started_server() {
    if [ "$STARTED_SERVER" -eq 1 ] && [ -n "$SERVER_PID" ]; then
        echo -e "${YELLOW}Stopping API started by this script (pid ${SERVER_PID}) ...${NC}"
        kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}

if [ "$PROMPT" -eq 1 ]; then
    trap stop_started_server EXIT
    if api_healthy; then
        echo -e "${GREEN}Reusing already-running API at ${BASE_URL}${NC}"
    else
        start_server_background
    fi

    PROMPT_ARGS=(
        --base-url "$BASE_URL"
        --mode "$MODE"
        --org-id "$ORG_ID"
        --user-id "$USER_ID"
    )
    if [ -n "$PATIENT_ID" ]; then
        PROMPT_ARGS+=(--patient-id "$PATIENT_ID")
    fi
    if [ -n "$QUESTION" ]; then
        PROMPT_ARGS+=(--question "$QUESTION")
    fi
    echo -e "${GREEN}Opening terminal prompt (mode=${MODE}) ...${NC}"
    python3 -m app.cli.prompt "${PROMPT_ARGS[@]}"
    exit $?
fi

echo -e "${GREEN}Starting RAG API server ...${NC}"
echo -e "${GREEN}API:              ${BASE_URL}${NC}"
echo -e "${GREEN}OpenAPI:          ${BASE_URL}/docs${NC}"
echo -e "${GREEN}Health:           ${BASE_URL}/health${NC}"
echo -e "${GREEN}Clinician UI:     ${BASE_URL}/app/${NC}"
echo -e "${GREEN}Full MedSwin:     POST ${BASE_URL}/api/v1/medswin/chat${NC}"
echo -e "${GREEN}Naive RAG:        POST ${BASE_URL}/api/v1/naive/chat${NC}"
echo -e "${GREEN}Side-by-side:     POST ${BASE_URL}/api/v1/naive/compare${NC}"
echo
echo -e "${YELLOW}In another terminal, prompt a question:${NC}"
echo -e "  ./scripts/start-local.sh --prompt"
echo -e "  ./scripts/start-local.sh --mode naive --question \"Your question\""
echo -e "  ./scripts/start-local.sh --mode both --question \"Your question\""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

python3 -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
