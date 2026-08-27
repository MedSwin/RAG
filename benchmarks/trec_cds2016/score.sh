#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export NIST_DIR="${NIST_DIR:-$ROOT/nist}"
python3 -m benchmarks.trec_cds2016.validate "$1"
python3 -m benchmarks.trec_cds2016.score "$1"
