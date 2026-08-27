#!/usr/bin/env python3
"""Automatic T4: full vs -gate vs -MAC action / facet / contradiction rates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.expert.schema import score_action
from benchmarks.trec_cds2016.stats import benjamini_hochberg, randomization_pvalue

PACKAGE_ROOT = Path(__file__).resolve().parent


def _load_packs(directory: Path, condition: str) -> list[dict]:
    packs = []
    for path in sorted(directory.glob(f"t3_medswin_*_{condition}_*.json")):
        packs.append(json.loads(path.read_text(encoding="utf-8")))
    if not packs:
        for path in sorted(directory.glob(f"t3_medswin_*.json")):
            pack = json.loads(path.read_text(encoding="utf-8"))
            if pack.get("condition", "full") == condition:
                packs.append(pack)
    return packs


def summarize_condition(packs: list[dict], condition: str) -> dict:
    actions = Counter()
    missing = Counter()
    contradictions = 0
    answered_by_topic: dict[int, int] = {}
    for pack in packs:
        answered = bool(pack.get("system_answered"))
        actions["answered" if answered else "abstained"] += 1
        answered_by_topic[int(pack["topic_id"])] = int(answered)
        sufficiency = pack.get("raw_sufficiency") or {}
        for facet in sufficiency.get("missing_facets") or []:
            missing[str(facet)] += 1
        contradictions += int(sufficiency.get("contradiction_count") or 0)
        gold = (pack.get("task_a") or {}).get("adjudicated")
        if gold:
            actions[score_action(answered, gold)] += 1
    n = max(len(packs), 1)
    return {
        "condition": condition,
        "n": len(packs),
        "answer_rate": actions.get("answered", 0) / n,
        "abstain_rate": actions.get("abstained", 0) / n,
        "missing_facets": dict(missing),
        "mean_contradictions": contradictions / n,
        "answered_by_topic": answered_by_topic,
    }


def _paired_p(left: dict[int, int], right: dict[int, int]) -> float:
    keys = sorted(set(left) & set(right))
    deltas = [float(right[key] - left[key]) for key in keys]
    return randomization_pvalue(deltas) if deltas else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs-dir", type=Path, default=PACKAGE_ROOT / "packs")
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "t4_automatic.json")
    args = parser.parse_args()
    conditions = {
        "full": _load_packs(args.packs_dir, "full"),
        "no_gate": _load_packs(args.packs_dir, "no_gate"),
        "no_mac": _load_packs(args.packs_dir, "no_mac"),
    }
    missing = [name for name, packs in conditions.items() if not packs]
    if missing:
        raise SystemExit(
            f"T4 packs missing for {', '.join(missing)} under {args.packs_dir}. "
            "Generate them with paper-eval --stage t4 before scoring."
        )
    rows = {name: summarize_condition(packs, name) for name, packs in conditions.items()}
    family: list[tuple[str, float]] = []
    if "full" in rows:
        for name in ("no_gate", "no_mac"):
            if name in rows:
                family.append(
                    (
                        f"full_vs_{name}_answer",
                        _paired_p(rows["full"]["answered_by_topic"], rows[name]["answered_by_topic"]),
                    )
                )
    for row in rows.values():
        row.pop("answered_by_topic", None)
    payload = {
        "confirmatory": False,
        "note": "Human T4 is not confirmatory. These are automatic process rates.",
        "conditions": rows,
        "fdr_family": [
            {"contrast": name, "p": pvalue, "q": qvalue, "star": star}
            for name, pvalue, qvalue, star in benjamini_hochberg(family)
        ]
        if family
        else [],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
