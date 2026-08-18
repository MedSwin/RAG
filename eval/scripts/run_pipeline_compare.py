#!/usr/bin/env python3
"""Run naive-RAG and full MedSwin on the same case file.

Usage (from the repository root, with the MedSwin API already running):

    python3 eval/scripts/run_pipeline_compare.py \
        --cases-path eval/data/sample/cases.jsonl \
        --max-cases 2

The script writes audit JSON under RUN_STORE_DIR (default /tmp/medswin-audits).
With the default `--pipeline both` that is two run files plus `{medswin_run_id}.comparison.json`.
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
    parser.add_argument("--pipeline", choices=("medswin", "naive_rag", "both"), default="both")
    parser.add_argument("--source-policy", choices=("ANY", "CPG_ONLY", "EMR_ONLY", "LIT_ONLY"), default="ANY")
    parser.add_argument("--clinical-scope", default="clinician_cds")
    parser.add_argument("--min-evidence-grade", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5, help="Dense top-K sent to /naive/chat; recorded on both run configs.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--reranker-budget", type=int, default=1)
    parser.add_argument("--ingest-case-context", action="store_true", default=True)
    parser.add_argument("--no-ingest-case-context", action="store_false", dest="ingest_case_context")
    args = parser.parse_args()

    settings = get_settings()
    request = RunRequest(
        cases_path=args.cases_path,
        max_cases=args.max_cases,
        source_policy=args.source_policy,
        clinical_scope=args.clinical_scope,
        min_evidence_grade=args.min_evidence_grade,
        top_k=args.top_k,
        max_concurrency=args.max_concurrency,
        reranker_budget=args.reranker_budget,
        ingest_case_context=args.ingest_case_context,
        pipeline=args.pipeline,
    )
    run = run_benchmark_sync(request, settings)
    comparison = (run.diagnostics or {}).get("pipeline_comparison") or {}
    print(f"pipeline={run.config.get('pipeline')}")
    print(f"run_id={run.run_id}")
    if comparison:
        print(f"naive_run_id={comparison.get('naive_run_id')}")
        print(f"medswin_run_id={comparison.get('medswin_run_id')}")
        print(f"delta={comparison.get('delta_medswin_minus_naive')}")
    else:
        print(f"aggregate={run.aggregate}")
    print(f"audits_dir={settings.run_store_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
