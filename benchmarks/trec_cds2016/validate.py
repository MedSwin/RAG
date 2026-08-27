#!/usr/bin/env python3
"""Validate a TREC CDS 2016 run file before official scoring."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .contract import RUN_DEPTH, TOPIC_IDS

_PMCID = re.compile(r"^[1-9][0-9]*$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9]{1,12}$")


def validate_run(path: Path, *, depth: int = RUN_DEPTH) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Run file missing: {path}")
    by_topic: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    run_names: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 6:
            raise RuntimeError(f"{path}:{line_no}: expected 6 columns, got {len(parts)}")
        qid_s, unused, doc_id, rank_s, score_s, run_name = parts
        if unused != "Q0":
            raise RuntimeError(f"{path}:{line_no}: second column must be Q0")
        try:
            qid = int(qid_s)
            rank = int(rank_s)
            score = float(score_s)
        except ValueError as exc:
            raise RuntimeError(f"{path}:{line_no}: bad numeric field") from exc
        if qid not in TOPIC_IDS:
            raise RuntimeError(f"{path}:{line_no}: qid {qid} not in 1-30")
        if not _PMCID.fullmatch(doc_id):
            raise RuntimeError(f"{path}:{line_no}: PMCID must be a bare integer, got {doc_id!r}")
        if not _RUN_NAME.fullmatch(run_name):
            raise RuntimeError(f"{path}:{line_no}: RUN_NAME must be 1-12 alphanumeric")
        run_names.add(run_name)
        by_topic[qid].append((rank, doc_id, score))

    missing = [qid for qid in TOPIC_IDS if qid not in by_topic]
    if missing:
        raise RuntimeError(f"{path}: missing topics {missing}")
    extra = sorted(set(by_topic) - set(TOPIC_IDS))
    if extra:
        raise RuntimeError(f"{path}: unexpected topics {extra}")
    if len(run_names) != 1:
        raise RuntimeError(f"{path}: expected one RUN_NAME, got {sorted(run_names)}")

    for qid in TOPIC_IDS:
        rows = by_topic[qid]
        if len(rows) > depth:
            raise RuntimeError(f"{path}: topic {qid} has {len(rows)} docs; max {depth}")
        docs = [doc_id for _rank, doc_id, _score in rows]
        if len(docs) != len(set(docs)):
            raise RuntimeError(f"{path}: topic {qid} has duplicate PMCIDs")
        ranks = [rank for rank, _doc, _score in rows]
        if ranks != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"{path}: topic {qid} ranks must be 1..N in order")
        scores = [score for _rank, _doc, score in rows]
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise RuntimeError(f"{path}: topic {qid} scores are not monotonically non-increasing")

    return {
        "path": str(path),
        "run_name": next(iter(run_names)),
        "topics": len(by_topic),
        "rows": sum(len(rows) for rows in by_topic.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = validate_run(args.run)
    print(f"OK {report['path']} run={report['run_name']} topics={report['topics']} rows={report['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
