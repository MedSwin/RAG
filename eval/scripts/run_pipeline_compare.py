#!/usr/bin/env python3
"""Run naive-RAG and full MedSwin on the same case file.

Usage (from the repository root, with the MedSwin API already running):

    python3 eval/scripts/run_pipeline_compare.py \
        --cases-path eval/data/sample/cases.jsonl \
        --max-cases 2

The script writes two audit JSON files plus a comparison JSON under RUN_STORE_DIR
(default /tmp/medswin-audits).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.app.config import get_settings
from eval.app.runner import run_benchmark_sync
from eval.app.schemas import RunRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare naive-RAG against full MedSwin.")
    parser.add_argument("--cases-path", default="eval/data/sample/cases.jsonl")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--source-policy", default="ANY")
    parser.add_argument("--clinical-scope", default="clinician_cds")
    parser.add_argument("--ingest-case-context", action="store_true", default=True)
    parser.add_argument("--no-ingest-case-context", action="store_false", dest="ingest_case_context")
    args = parser.parse_args()

    settings = get_settings()
    request = RunRequest(
        cases_path=args.cases_path,
        max_cases=args.max_cases,
        source_policy=args.source_policy,
        clinical_scope=args.clinical_scope,
        ingest_case_context=args.ingest_case_context,
        pipeline="both",
    )
    run = run_benchmark_sync(request, settings)
    comparison = (run.diagnostics or {}).get("pipeline_comparison") or {}
    print(f"medswin_run_id={run.run_id}")
    print(f"naive_run_id={comparison.get('naive_run_id')}")
    print(f"delta={comparison.get('delta_medswin_minus_naive')}")
    print(f"audits_dir={settings.run_store_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
