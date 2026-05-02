#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-app}"

if [ ! -d "$ROOT_DIR" ]; then
  echo "Directory \"$ROOT_DIR\" does not exist" >&2
  exit 1
fi

echo "Searching for cache directories in \"$ROOT_DIR\"..."

CACHE_DIRS=$(find "$ROOT_DIR" -type d \( -name "__pycache__" -o -name ".pytest_cache" \))

if [ -z "$CACHE_DIRS" ]; then
  echo "No __pycache__ or .pytest_cache directories found."
  exit 0
fi

while IFS= read -r dir; do
  echo "Removing $dir"
  rm -rf "$dir"
done <<< "$CACHE_DIRS"
