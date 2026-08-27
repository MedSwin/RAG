#!/bin/bash

# MedSwin local operator.
# A TTY session becomes the interactive console. Non-interactive use stays
# `serve`. Publication evaluation is `paper-eval`. Ordinary serve/console/ask
# do not download the TREC corpus.

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
CLI_PORT=""
PORTAL="clinician"
OPEN_BROWSER=""
JSON=0
SKIP_EVAL_WARMUP=0
FORCE_EVAL_WARMUP=0
RESET_FULL_CORPUS=0
STAGE="all"
SYSTEMS="all"
PAPER_PIPELINE=""
GENERATOR="cloud"
GENERATOR_SET=0
TOPIC_FIELD="note"
ALLOW_LOCAL_T3=0
WITH_LOCAL_LLM=0
WITH_COMPOSE_API=0
STOP_MONGO=0

usage() {
    cat <<'EOF'
Usage:
  ./scripts/start-local.sh [command] [options]

Commands
  console     Interactive operator (default in a terminal)
  serve       API only, foreground
  up          Start API, print portals, open the clinician UI
  ask         Prompt full MedSwin, naive-RAG, or both
  paper-eval  Official TREC CDS 2016 + T3/T4 publication path
  warmup      Download/verify NIST files, trec_eval, ir_datasets, optional HF/Foundry
  full-eval   Deprecated alias for paper-eval --stage all
  eval        Removed; prints the replacement and exits 1
  open        Open a portal in the browser
  status      API / corpus / naive / paper-eval health
  index       Refresh embeddings and rebuild the ANN index
  stop        Stop API / local-LLM processes started by this operator

Default
  Terminal (TTY)     → console
  Non-interactive    → serve

Ask options
  --mode MODE           full | naive | both
  --question TEXT       One-shot question (implies ask)
  --prompt              Alias for ask
  --org-id ID           Tenant org_id                (default: demo-org)
  --user-id ID          user_id                      (default: clinician-1)
  --patient-id ID       Optional EMR scope
  --port N              API port                     (default: 8100)
  --portal NAME         clinician|dashboard|docs|all
  --open / --no-open    Open browsers
  --json                Machine-readable operator output

paper-eval options
  --stage S             warmup|prepare|emit|score|t3|t4|all
  --systems LIST        bm25,dense,rrf,cascade or all     (T1)
  --pipeline P          naive|medswin|both                (T3/T4)
  --generator G         cloud|medswin|both                (T3 packs)
  --topic-field F       note|summary                      (T1)
  --reset-full-corpus   Rebuild the isolated 1.25M corpus
  --force-eval-warmup   Redownload warmup artifacts
  --skip-eval-warmup    Skip warmup slices (ignored for score)
  --allow-local-t3      Permit 7B human T3 packs (not the paper freeze)
  --with-local-llm      Download/serve MedSwin 7B
  --with-compose-api    Start rag_api from Compose with benchmark mounts
  --stop-mongo          On stop, also docker compose stop mongodb

Examples
  ./scripts/start-local.sh warmup
  ./scripts/start-local.sh paper-eval
  ./scripts/start-local.sh paper-eval --systems bm25,dense
  ./scripts/start-local.sh paper-eval --systems rrf,cascade --topic-field note
  ./scripts/start-local.sh paper-eval --stage emit --systems cascade
  ./scripts/start-local.sh paper-eval --stage score
  ./scripts/start-local.sh paper-eval --pipeline both --generator cloud --stage t3
  ./scripts/start-local.sh paper-eval --pipeline medswin --stage t4
  ./scripts/start-local.sh paper-eval --reset-full-corpus
  ./scripts/start-local.sh ask --mode both --question "..."
EOF
}

KNOWN_COMMANDS="console serve up ask prompt eval warmup full-eval paper-eval open status index stop"

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
        --portal) PORTAL="${2:-clinician}"; shift 2 ;;
        --open) OPEN_BROWSER="1"; shift ;;
        --no-open) OPEN_BROWSER="0"; shift ;;
        --skip-eval-warmup) SKIP_EVAL_WARMUP=1; shift ;;
        --force-eval-warmup) FORCE_EVAL_WARMUP=1; shift ;;
        --reset-full-corpus) RESET_FULL_CORPUS=1; shift ;;
        --stage) STAGE="${2:-all}"; shift 2 ;;
        --systems) SYSTEMS="${2:-all}"; shift 2 ;;
        --pipeline) PAPER_PIPELINE="${2:-}"; shift 2 ;;
        --generator) GENERATOR="${2:-cloud}"; GENERATOR_SET=1; shift 2 ;;
        --topic-field) TOPIC_FIELD="${2:-note}"; shift 2 ;;
        --allow-local-t3) ALLOW_LOCAL_T3=1; shift ;;
        --with-local-llm) WITH_LOCAL_LLM=1; shift ;;
        --with-compose-api) WITH_COMPOSE_API=1; shift ;;
        --stop-mongo) STOP_MONGO=1; shift ;;
        --json) JSON=1; shift ;;
        --eval-port|--with-eval|--run|--cases-path|--max-cases|--top-k)
            echo -e "${RED}Removed publication flag: $1. Use paper-eval.${NC}"
            usage
            exit 1
            ;;
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
if [[ -n "$PAPER_PIPELINE" && "$PAPER_PIPELINE" != "naive" && "$PAPER_PIPELINE" != "medswin" && "$PAPER_PIPELINE" != "both" ]]; then
    echo -e "${RED}--pipeline must be naive, medswin, or both${NC}"
    exit 1
fi
if [[ "$GENERATOR" != "cloud" && "$GENERATOR" != "medswin" && "$GENERATOR" != "both" ]]; then
    echo -e "${RED}--generator must be cloud, medswin, or both${NC}"
    exit 1
fi
if [[ "$TOPIC_FIELD" != "note" && "$TOPIC_FIELD" != "summary" ]]; then
    echo -e "${RED}--topic-field must be note or summary${NC}"
    exit 1
fi
if [[ "$STAGE" != "warmup" && "$STAGE" != "prepare" && "$STAGE" != "emit" && "$STAGE" != "score" && "$STAGE" != "t3" && "$STAGE" != "t4" && "$STAGE" != "all" ]]; then
    echo -e "${RED}--stage must be warmup|prepare|emit|score|t3|t4|all${NC}"
    exit 1
fi

reject_illegal_paper_eval() {
    if [[ "$STAGE" == "t3" || "$STAGE" == "t4" ]]; then
        if [[ "$SYSTEMS" != "all" ]]; then
            echo -e "${RED}--systems applies to T1 only. Do not pass it with --stage $STAGE.${NC}"
            exit 1
        fi
    fi
    if [[ "$STAGE" == "emit" || "$STAGE" == "score" ]]; then
        if [[ -n "$PAPER_PIPELINE" || "$GENERATOR_SET" -eq 1 ]]; then
            echo -e "${RED}--pipeline / --generator apply to T3/T4 only. T1 emit/score has no generator.${NC}"
            exit 1
        fi
    fi
    if [[ "$STAGE" == "t3" && ( "$GENERATOR" == "medswin" || "$GENERATOR" == "both" ) && "$ALLOW_LOCAL_T3" -eq 0 ]]; then
        echo -e "${RED}Human T3 freezes Foundry GPT. Pass --allow-local-t3 only for exploratory 7B packs.${NC}"
        exit 1
    fi
}

if [[ "$COMMAND" == "paper-eval" || "$COMMAND" == "full-eval" ]]; then
    reject_illegal_paper_eval
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
BASE_URL="${MEDSWIN_BASE_URL:-http://127.0.0.1:${APP_PORT}}"

export FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5.4}"
export CLOUD_MODEL="${CLOUD_MODEL:-$FOUNDRY_MODEL}"
export CLOUD_EMBEDDING="${CLOUD_EMBEDDING:-embed-v-4-0}"
export CLOUD_EMBEDDING_DIMENSION="${CLOUD_EMBEDDING_DIMENSION:-1536}"
export CLOUD_RERANKER="${CLOUD_RERANKER:-Cohere-rerank-v4.0-fast}"
export CLOUD_MODEL_INFERENCE_API_VERSION="${CLOUD_MODEL_INFERENCE_API_VERSION:-2024-05-01-preview}"
export CLOUD_RERANKER_AUTH_SCHEME="${CLOUD_RERANKER_AUTH_SCHEME:-bearer}"
export MEDSWIN_MODEL_REPO="${MEDSWIN_MODEL_REPO:-MedSwin/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_MODEL_PATH="${MEDSWIN_MODEL_PATH:-./models/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_LLM_MODEL="${MEDSWIN_LLM_MODEL:-MedSwin/MedSwin-DaRE-TIES-KD-0.7}"
export MEDSWIN_LLM_URL="${MEDSWIN_LLM_URL:-http://127.0.0.1:8000/v1/chat/completions}"
export EVAL_WARMUP_ON_START="${EVAL_WARMUP_ON_START:-false}"
export BENCHMARK_ORG_ID="${BENCHMARK_ORG_ID:-bench-org}"

need_full_bootstrap() {
    case "$COMMAND" in
        stop|status|open|eval) return 1 ;;
        *) return 0 ;;
    esac
}

bind_benchmark_index() {
    local eval_data_dir="${FULL_EVAL_DATA_DIR:-./data/full-trec-benchmark}"
    mkdir -p "$eval_data_dir"
    export HNSW_INDEX_PATH="${FULL_EVAL_HNSW_INDEX_PATH:-${eval_data_dir}/hnsw_index.bin}"
    export HNSW_MAPPING_PATH="${FULL_EVAL_HNSW_MAPPING_PATH:-${eval_data_dir}/hnsw_mapping.sqlite}"
    export FAISS_INDEX_PATH="${FULL_EVAL_FAISS_INDEX_PATH:-${eval_data_dir}/faiss_unused.bin}"
    export FAISS_MAPPING_PATH="${FULL_EVAL_FAISS_MAPPING_PATH:-${eval_data_dir}/faiss_unused.json}"
    export TREE_INDEX_PATH="${FULL_EVAL_TREE_INDEX_PATH:-${eval_data_dir}/tree_unused.npy}"
    export TREE_MAPPING_PATH="${FULL_EVAL_TREE_MAPPING_PATH:-${eval_data_dir}/tree_unused.json}"
    export LEXICAL_FTS_PATH="${FULL_EVAL_LEXICAL_FTS_PATH:-${eval_data_dir}/bm25.sqlite}"
    export LLM_TIMEOUT_S="${FULL_EVAL_LLM_TIMEOUT_S:-600}"
    export CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE="${CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE:-query}"
}

ensure_mongo() {
    export MONGODB_URL="${MONGODB_URL:-mongodb://localhost:27017}"
    export MONGODB_DB="${MONGODB_DB:-medswin}"
    export MONGODB_DATABASE="${MONGODB_DATABASE:-medswin}"
    if python3 -c "import os, pymongo; pymongo.MongoClient(os.environ['MONGODB_URL']).admin.command('ping')" >/dev/null 2>&1; then
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}MongoDB is not reachable and Docker is not installed.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Starting Compose Mongo (medswin-mongodb / mongo:7.0) ...${NC}"
    docker compose up -d mongodb
    local tries=0
    export MONGODB_URL="mongodb://${MONGO_ROOT_USERNAME:-admin}:${MONGO_ROOT_PASSWORD:-password123}@127.0.0.1:27017/medswin?authSource=admin"
    while [[ "$tries" -lt 40 ]]; do
        if python3 -c "import os, pymongo; pymongo.MongoClient(os.environ['MONGODB_URL']).admin.command('ping')" >/dev/null 2>&1; then
            echo -e "${GREEN}Compose Mongo is healthy.${NC}"
            return 0
        fi
        sleep 2
        tries=$((tries + 1))
    done
    echo -e "${RED}Compose Mongo did not become healthy.${NC}"
    exit 1
}

run_eval_warmup() {
    local args=()
    if [[ "$FORCE_EVAL_WARMUP" -eq 1 ]]; then args+=(--force); fi
    if [[ "$WITH_LOCAL_LLM" -eq 1 || "$GENERATOR" == "medswin" || "$GENERATOR" == "both" ]]; then
        export PAPER_EVAL_NEED_LOCAL_LLM=1
        args+=(--with-local-llm)
    else
        args+=(--skip-model)
    fi
    if [[ "$SYSTEMS" == "bm25" && -z "$PAPER_PIPELINE" && "$STAGE" != "t3" && "$STAGE" != "t4" ]]; then
        args+=(--skip-foundry)
    fi
    echo -e "${YELLOW}Paper-eval warmup: NIST + trec_eval + ir_datasets ...${NC}"
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

    ensure_mongo

    mkdir -p models data logs storage
    export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-./models/MedEmbed-large-v0.1}"
    export RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-./models/bge-reranker-v2-m3}"
    export HNSW_INDEX_PATH="${HNSW_INDEX_PATH:-./data/hnsw_index.bin}"
    export HNSW_MAPPING_PATH="${HNSW_MAPPING_PATH:-./data/hnsw_mapping.json}"
    export DATA_DIR="${DATA_DIR:-./data}"
    export DEBUG="${DEBUG:-true}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export APP_HOST
    export APP_PORT

    if [[ "$COMMAND" != "warmup" && "$COMMAND" != "paper-eval" && "$COMMAND" != "full-eval" && "$SKIP_EVAL_WARMUP" -eq 0 && "$EVAL_WARMUP_ON_START" == "true" ]]; then
        run_eval_warmup
    fi
fi

operator() {
    local extra=()
    extra+=(--base-url "$BASE_URL")
    extra+=(--host "$APP_HOST" --port "$APP_PORT")
    extra+=(--mode "$MODE" --org-id "$ORG_ID" --user-id "$USER_ID")
    extra+=(--portal "$PORTAL")
    if [ -n "$PATIENT_ID" ]; then extra+=(--patient-id "$PATIENT_ID"); fi
    if [ -n "$QUESTION" ]; then extra+=(--question "$QUESTION"); fi
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
    echo
    echo -e "${YELLOW}In another terminal:${NC}"
    echo -e "  ./scripts/start-local.sh ask --mode both"
    echo -e "  ./scripts/start-local.sh paper-eval"
    echo -e "  ./scripts/start-local.sh status"
}

paper_prepare() {
    local bench_org="${BENCHMARK_ORG_ID:-bench-org}"
    local checkpoint="data/eval-warmup/full-trec-${bench_org}-checkpoint.json"
    local prep_args=(--org-id "$bench_org")
    bind_benchmark_index
    echo -e "${YELLOW}Preparing tenant-safe Mongo indexes for complete TREC ingestion ...${NC}"
    CLOUD_MODE=true python3 -m benchmarks.trec_cds2016.prepare.mongo --org-id "$bench_org"
    if [[ "$RESET_FULL_CORPUS" -eq 1 || ! -f "$checkpoint" ]]; then
        prep_args+=(--reset)
    fi
    echo -e "${YELLOW}Preparing complete TREC-CDS runtime corpus and indexes ...${NC}"
    CLOUD_MODE=true python3 -m benchmarks.trec_cds2016.prepare.runtime "${prep_args[@]}"
    echo -e "${YELLOW}Materializing complete TREC document metadata layer ...${NC}"
    if [[ "$RESET_FULL_CORPUS" -eq 1 ]]; then
        CLOUD_MODE=true python3 -m benchmarks.trec_cds2016.prepare.materialize --org-id "$bench_org" --force
    else
        CLOUD_MODE=true python3 -m benchmarks.trec_cds2016.prepare.materialize --org-id "$bench_org"
    fi
    echo -e "${YELLOW}Verifying persisted 100% TREC documents, chunks, embeddings, BM25 and HNSW ...${NC}"
    CLOUD_MODE=true python3 -m benchmarks.trec_cds2016.prepare.verify --org-id "$bench_org"
}

paper_emit() {
    bind_benchmark_index
    CLOUD_MODE=true CLOUD_EMBEDDING_DEFAULT_INPUT_TYPE=query \
        python3 -m benchmarks.trec_cds2016.emit \
        --systems "$SYSTEMS" \
        --topic-field "$TOPIC_FIELD" \
        --org-id "${BENCHMARK_ORG_ID:-bench-org}"
}

paper_score() {
    python3 - <<'PY'
from benchmarks.trec_cds2016.contract import PACKAGE_ROOT, RUN_NAMES, SUMMARY_RUN_NAMES
from benchmarks.trec_cds2016.nist import verify_nist
from benchmarks.trec_cds2016.score import score_run
from benchmarks.trec_cds2016.contrast import load_systems
from benchmarks.trec_cds2016.stats import contrast_infndcg
import json, os
verify_nist()
field = os.environ.get("PAPER_TOPIC_FIELD", "note")
names = SUMMARY_RUN_NAMES if field == "summary" else RUN_NAMES
runs_dir = PACKAGE_ROOT / "runs"
scores_dir = PACKAGE_ROOT / "scores"
written = []
for path in sorted(runs_dir.glob("*.run")):
    if path.stem not in names.values():
        continue
    score_run(path, scores_dir=scores_dir)
    written.append(path.stem)
    print(f"scored {path.name}")
if not written:
    raise SystemExit("No T1 run files found under benchmarks/trec_cds2016/runs/")
per_system = load_systems(scores_dir, field)
if all(name in per_system for name in ("bm25", "dense", "rrf", "cascade")):
    payload = contrast_infndcg(per_system)
    out = scores_dir / f"contrasts_{field}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
PY
}

ensure_paper_api() {
    bind_benchmark_index
    local api_port="${FULL_EVAL_API_PORT:-8110}"
    export MEDSWIN_BASE_URL="http://127.0.0.1:${api_port}"
    if [[ "$WITH_COMPOSE_API" -eq 1 ]]; then
        docker compose up -d rag_api
        return 0
    fi
    python3 - <<PY
from app.cli.surfaces import ensure_api
ensure_api("http://127.0.0.1:${api_port}", host="0.0.0.0", port=${api_port})
print("paper API ready on ${api_port}")
PY
}

ensure_local_llm() {
    python3 - <<'PY'
import os, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen
root = Path(".").resolve()
log_dir = root / "logs"
log_dir.mkdir(exist_ok=True)
pid_path = log_dir / "medswin-llm.pid"
url = os.environ.get("MEDSWIN_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
health = url.rsplit("/v1/", 1)[0] + "/health"
try:
    urlopen(health, timeout=2)
    print("local 7B reused")
    raise SystemExit(0)
except Exception:
    pass
handle = (log_dir / "medswin-llm.log").open("a")
proc = subprocess.Popen(
    [sys.executable, str(root / "scripts" / "serve-medswin-llm.py")],
    cwd=str(root),
    stdout=handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
pid_path.write_text(str(proc.pid), encoding="utf-8")
deadline = time.time() + 900
while time.time() < deadline:
    try:
        urlopen(health, timeout=2)
        print(f"local 7B started pid={proc.pid}")
        raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit("local 7B did not become healthy")
PY
}

paper_t3() {
    ensure_paper_api
    if [[ "$GENERATOR" == "medswin" || "$GENERATOR" == "both" || "$WITH_LOCAL_LLM" -eq 1 ]]; then
        ensure_local_llm
    fi
    local pipelines="$PAPER_PIPELINE"
    if [[ -z "$pipelines" ]]; then
        pipelines="both"
    fi
    local generators="$GENERATOR"
    local extra=()
    if [[ "$ALLOW_LOCAL_T3" -eq 1 ]]; then
        extra+=(--allow-local-t3)
    fi
    if [[ "$generators" == "both" ]]; then
        CLOUD_MODE=true python3 -m benchmarks.expert.t3_packs --pipeline "$pipelines" --generator cloud --condition full "${extra[@]}" --base-url "${MEDSWIN_BASE_URL}"
        python3 -m benchmarks.expert.t3_packs --pipeline "$pipelines" --generator medswin --condition full --allow-local-t3 --base-url "${MEDSWIN_BASE_URL}"
    else
        CLOUD_MODE=true python3 -m benchmarks.expert.t3_packs --pipeline "$pipelines" --generator "$generators" --condition full "${extra[@]}" --base-url "${MEDSWIN_BASE_URL}"
    fi
}

paper_t4() {
    ensure_paper_api
    local extra=()
    if [[ "$ALLOW_LOCAL_T3" -eq 1 ]]; then extra+=(--allow-local-t3); fi
    for condition in full no_gate no_mac; do
        CLOUD_MODE=true python3 -m benchmarks.expert.t3_packs \
            --pipeline medswin \
            --generator cloud \
            --condition "$condition" \
            "${extra[@]}" \
            --base-url "${MEDSWIN_BASE_URL}"
    done
    python3 -m benchmarks.expert.t4_automatic
}

run_paper_eval() {
    reject_illegal_paper_eval
    export PAPER_TOPIC_FIELD="$TOPIC_FIELD"
    local stages=()
    case "$STAGE" in
        all)
            stages=(warmup prepare emit score)
            if [[ -n "$PAPER_PIPELINE" ]]; then
                stages+=(t3)
            fi
            ;;
        *) stages=("$STAGE") ;;
    esac
    for step in "${stages[@]}"; do
        case "$step" in
            warmup)
                if [[ "$SKIP_EVAL_WARMUP" -eq 1 ]]; then
                    echo -e "${YELLOW}Skipping warmup slices (score still requires NIST files).${NC}"
                else
                    run_eval_warmup
                fi
                ;;
            prepare) paper_prepare ;;
            emit) paper_emit ;;
            score)
                if [[ "$SKIP_EVAL_WARMUP" -eq 1 ]]; then
                    python3 -c "from benchmarks.trec_cds2016.nist import verify_nist; verify_nist()"
                fi
                paper_score
                ;;
            t3) paper_t3 ;;
            t4) paper_t4 ;;
        esac
    done
}

cmd_eval_removed() {
    echo -e "${RED}The MSAS eval portal is removed.${NC}"
    echo "Use: ./scripts/start-local.sh paper-eval"
    echo "T1 default: ./scripts/start-local.sh paper-eval --stage all"
    echo "T3 packs:   ./scripts/start-local.sh paper-eval --pipeline both --generator cloud --stage t3"
    exit 1
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
    eval) cmd_eval_removed ;;
    warmup) run_eval_warmup ;;
    paper-eval) run_paper_eval ;;
    full-eval)
        echo -e "${YELLOW}full-eval is deprecated. Running: ./scripts/start-local.sh paper-eval --stage all${NC}"
        STAGE="all"
        run_paper_eval
        ;;
    open) operator open ;;
    status) operator status ;;
    index) operator index ;;
    stop)
        extra=()
        if [[ "$STOP_MONGO" -eq 1 ]]; then extra+=(--stop-mongo); fi
        operator stop "${extra[@]}"
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac
