#!/usr/bin/env python3
"""Score the three pre-registered infNDCG contrasts from official score JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import PACKAGE_ROOT, RUN_NAMES, SUMMARY_RUN_NAMES
from .stats import contrast_infndcg

SCORES_DIR = PACKAGE_ROOT / "scores"


def _per_topic(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metrics"]["infNDCG"]["per_topic"]


def load_systems(scores_dir: Path, field: str = "note") -> dict[str, dict[str, float]]:
    names = SUMMARY_RUN_NAMES if field == "summary" else RUN_NAMES
    per_system: dict[str, dict[str, float]] = {}
    for system, run_name in names.items():
        path = scores_dir / f"{run_name}.json"
        if path.is_file():
            per_system[system] = _per_topic(path)
    return per_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--topic-field", choices=("note", "summary"), default="note")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    per_system = load_systems(args.scores_dir, args.topic_field)
    missing = [name for name in ("bm25", "dense", "rrf", "cascade") if name not in per_system]
    if missing:
        raise SystemExit(f"Missing official score JSON for {missing}")
    payload = contrast_infndcg(per_system)
    out = args.out or args.scores_dir / f"contrasts_{args.topic_field}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
