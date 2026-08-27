#!/usr/bin/env python3
"""Score a validated run with official NIST tools only."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from .contract import DIAGNOSIS_TOPICS, PACKAGE_ROOT, TABLE8_NOTE_MEDIAN_INFNDCG, TABLE8_NOTE_MEDIAN_P10, TEST_TOPICS, TREATMENT_TOPICS
from .nist import ensure_trec_eval, nist_paths
from .validate import validate_run

SCORES_DIR = PACKAGE_ROOT / "scores"

_MEASURE = re.compile(r"^(\S+)\s+(\S+)\s+([-+0-9.eE]+)\s*$")


def _parse_trec_output(text: str) -> dict[str, dict[str, float]]:
    by_topic: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _MEASURE.match(line.strip())
        if not match:
            continue
        measure, qid, value = match.group(1), match.group(2), float(match.group(3))
        by_topic.setdefault(qid, {})[measure] = value
    return by_topic


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _topic_values(by_topic: dict[str, dict[str, float]], measure: str, topics: tuple[int, ...]) -> list[float]:
    return [by_topic[str(qid)][measure] for qid in topics if str(qid) in by_topic and measure in by_topic[str(qid)]]


def score_run(run_path: Path, *, scores_dir: Path | None = None) -> dict[str, object]:
    validate_run(run_path)
    paths = nist_paths()
    if paths["qrels-sampleval-2016.txt"].name == "qrels-treceval-2016.txt":
        raise RuntimeError("Refusing to score inferred measures with trec_eval qrels")
    sample_eval = paths["sample_eval.pl"]
    treceval_qrels = paths["qrels-treceval-2016.txt"]
    sample_qrels = paths["qrels-sampleval-2016.txt"]
    trec_eval = ensure_trec_eval()

    inferred = subprocess.run(
        ["perl", str(sample_eval), "-q", str(sample_qrels), str(run_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    residual = subprocess.run(
        [str(trec_eval), "-q", "-c", "-M1000", str(treceval_qrels), str(run_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    inferred_topics = _parse_trec_output(inferred.stdout)
    residual_topics = _parse_trec_output(residual.stdout)

    judged = _parse_qrels(treceval_qrels)
    unjudged_at_10 = _unjudged_at_k(run_path, judged, k=10)

    def pack(measure: str, source: dict[str, dict[str, float]]) -> dict[str, object]:
        all_topics = tuple(range(1, 31))
        return {
            "all": _mean(_topic_values(source, measure, all_topics)),
            "diagnosis": _mean(_topic_values(source, measure, DIAGNOSIS_TOPICS)),
            "test": _mean(_topic_values(source, measure, TEST_TOPICS)),
            "treatment": _mean(_topic_values(source, measure, TREATMENT_TOPICS)),
            "per_topic": {qid: source[qid][measure] for qid in source if measure in source[qid] and qid != "all"},
        }

    payload = {
        "run": str(run_path),
        "tools": {
            "sample_eval": str(sample_eval),
            "trec_eval": str(trec_eval),
            "qrels_treceval": str(treceval_qrels),
            "qrels_sampleval": str(sample_qrels),
        },
        "metrics": {
            "infNDCG": pack("infNDCG", inferred_topics),
            "infAP": pack("infAP", inferred_topics),
            "P_10": pack("P_10", residual_topics),
            "Rprec": pack("Rprec", residual_topics),
        },
        "diagnostics": {
            "unjudged_at_10_mean": _mean(list(unjudged_at_10.values())),
            "unjudged_at_10": unjudged_at_10,
            "do_not_report_iP10_as_official_P10": True,
        },
        "historical_note_table8": {
            "median_infNDCG": TABLE8_NOTE_MEDIAN_INFNDCG,
            "median_P_10": TABLE8_NOTE_MEDIAN_P10,
            "note": "Field-matched automatic note medians. Not Table 6 (summary best-per-team).",
        },
    }
    out_dir = scores_dir or SCORES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_qrels(path: Path) -> dict[int, set[str]]:
    judged: dict[int, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _it, doc_id, _rel = parts[0], parts[1], parts[2], parts[3]
        judged.setdefault(int(qid), set()).add(doc_id)
    return judged


def _unjudged_at_k(run_path: Path, judged: dict[int, set[str]], k: int) -> dict[str, float]:
    rates: dict[str, float] = {}
    by_topic: dict[int, list[str]] = {}
    for line in run_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        qid, doc_id = int(parts[0]), parts[2]
        by_topic.setdefault(qid, []).append(doc_id)
    for qid, docs in by_topic.items():
        head = docs[:k]
        gold = judged.get(qid, set())
        rates[str(qid)] = sum(1 for doc in head if doc not in gold) / max(len(head), 1)
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = score_run(args.run, scores_dir=args.out_dir)
    metrics = payload["metrics"]
    print(
        f"{args.run.name} infNDCG={metrics['infNDCG']['all']:.4f} "
        f"infAP={metrics['infAP']['all']:.4f} "
        f"P@10={metrics['P_10']['all']:.4f} "
        f"Rprec={metrics['Rprec']['all']:.4f} "
        f"unjudged@10={payload['diagnostics']['unjudged_at_10_mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
