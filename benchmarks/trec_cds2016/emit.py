#!/usr/bin/env python3
"""Emit official TREC run files for pre-registered T1 systems."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from app.core.database import close_database, init_database

from .contract import PACKAGE_ROOT, RUN_NAMES, SUMMARY_RUN_NAMES, SYSTEMS
from .exporter import LitOnlyExporter, write_run
from .topics import TopicField, load_topics
from .validate import validate_run

RUNS_DIR = PACKAGE_ROOT / "runs"


async def emit_systems(
    systems: list[str],
    *,
    org_id: str,
    field: TopicField,
    runs_dir: Path,
) -> list[Path]:
    await init_database()
    exporter = LitOnlyExporter()
    written: list[Path] = []
    try:
        topics = load_topics()
        names = SUMMARY_RUN_NAMES if field == "summary" else RUN_NAMES
        for system in systems:
            if system not in SYSTEMS:
                raise ValueError(f"Unknown system {system}; expected one of {SYSTEMS}")
            run_name = names[system]
            rankings = []
            for topic in topics:
                ranked = await exporter.retrieve_system(topic, system, org_id=org_id, field=field)
                rankings.append((topic.number, ranked))
                print(f"{system} topic {topic.number} docs={len(ranked)}", flush=True)
            path = runs_dir / f"{run_name}.run"
            write_run(path, run_name, rankings)
            validate_run(path)
            written.append(path)
            print(f"wrote {path}", flush=True)
    finally:
        await exporter.close()
        await close_database()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="all", help="comma list: bm25,dense,rrf,cascade or all")
    parser.add_argument("--topic-field", choices=("note", "summary"), default="note")
    parser.add_argument("--org-id", default=os.environ.get("BENCHMARK_ORG_ID", "bench-org"))
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    systems = SYSTEMS if args.systems == "all" else tuple(item.strip() for item in args.systems.split(",") if item.strip())
    asyncio.run(emit_systems(list(systems), org_id=args.org_id, field=args.topic_field, runs_dir=args.runs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
