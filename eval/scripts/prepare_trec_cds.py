#!/usr/bin/env python3
"""Prepare TREC CDS 2016 benchmark cases for the MedSwin system audit.

This script uses ir_datasets to export a compact JSONL case file from
`pmc/v2/trec-cds-2016`. It does not download data itself beyond what
ir_datasets requires. You must have legitimate access to the TREC CDS/MIMIC
resources required by the dataset loader.

Example:
  python scripts/prepare_trec_cds.py \
    --dataset pmc/v2/trec-cds-2016 \
    --out data/trec_cds_2016/cases.jsonl \
    --max-topics 30 \
    --max-docs-per-topic 200
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ir_datasets

# Root Cause vs Logic: this script is commonly run as `python3 scripts/...`
# from either the `eval/` directory or the repo root, so `eval/` is not always
# on `sys.path`. We add the package root explicitly before importing `app`.
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from app.io import write_jsonl


QUERY_TYPE_TO_FACETS = {
    "diagnosis": [
        {"facet_id": "dx", "name": "diagnosis evidence", "weight": 1.0, "critical": True},
        {"facet_id": "patient_fit", "name": "patient applicability", "weight": 1.0, "critical": True},
    ],
    "test": [
        {"facet_id": "test_indication", "name": "test indication", "weight": 1.0, "critical": True},
        {"facet_id": "patient_fit", "name": "patient applicability", "weight": 1.0, "critical": True},
        {"facet_id": "risk", "name": "test risks or limitations", "weight": 0.75, "critical": False},
    ],
    "treatment": [
        {"facet_id": "treatment", "name": "treatment recommendation evidence", "weight": 1.0, "critical": True},
        {"facet_id": "safety", "name": "safety contraindications or adverse risks", "weight": 1.2, "critical": True},
        {"facet_id": "patient_fit", "name": "patient applicability", "weight": 1.0, "critical": True},
    ],
}


def topic_text(query: Any) -> str:
    # Motivation vs Logic: the benchmark query must preserve the original TREC CDS
    # clinical information need so retrieval quality reflects evidence matching,
    # not a synthetic placeholder that collapses all topics to the same prompt.
    description = getattr(query, "description", None) or ""
    summary = getattr(query, "summary", None) or ""
    note = getattr(query, "note", None) or ""
    parts = [str(part).strip() for part in (description, summary, note) if str(part).strip()]
    return parts[0] if parts else f"Clinical question for topic {getattr(query, 'query_id', '')}"


def get_patient_context(query: Any) -> str:
    # TREC CDS query fields vary by track. In ir_datasets 2016, fields usually
    # include description and summary. If a note field exists, prefer it.
    for name in ("note", "description", "summary"):
        value = getattr(query, name, None)
        if value:
            return str(value)
    return ""


def select_gold_docs(
    qid: str,
    qrels_by_qid: dict[str, list[str]],
    relevance_by_qid: dict[str, dict[str, int]],
    max_docs_per_topic: int | None,
) -> list[str]:
    docs = list(dict.fromkeys(qrels_by_qid.get(qid, [])))
    # Motivation vs Logic: TREC CDS runs should stay reproducible, so we cap
    # evidence deterministically rather than sampling randomly. We prefer higher
    # relevance grades first and use doc_id as a stable tiebreaker.
    if max_docs_per_topic is None or len(docs) <= max_docs_per_topic:
        return sorted(docs)
    return sorted(
        docs,
        key=lambda doc_id: (
            -int(relevance_by_qid.get(qid, {}).get(doc_id, 0)),
            doc_id,
        ),
    )[:max_docs_per_topic]


def prepare(dataset_name: str, out_path: str, max_topics: int | None, max_docs_per_topic: int | None) -> None:
    dataset = ir_datasets.load(dataset_name)
    qrels_by_qid: dict[str, list[str]] = defaultdict(list)
    relevance_by_qid: dict[str, dict[str, int]] = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        # TREC qrels use relevance grades; treat >0 as evidence-relevant.
        if int(qrel.relevance) > 0:
            qrels_by_qid[str(qrel.query_id)].append(str(qrel.doc_id))
            relevance_by_qid[str(qrel.query_id)][str(qrel.doc_id)] = int(qrel.relevance)

    rows = []
    for idx, query in enumerate(dataset.queries_iter()):
        if max_topics is not None and idx >= max_topics:
            break
        qid = str(getattr(query, "query_id"))
        qtype = str(getattr(query, "type", "clinical")).lower()
        patient_context = get_patient_context(query)
        gold_docs = select_gold_docs(qid, qrels_by_qid, relevance_by_qid, max_docs_per_topic)
        facets = QUERY_TYPE_TO_FACETS.get(qtype, [
            {"facet_id": "clinical_evidence", "name": "clinically relevant evidence", "weight": 1.0, "critical": True},
            {"facet_id": "patient_fit", "name": "patient applicability", "weight": 1.0, "critical": True},
        ])
        # Attach relevant doc ids to all generated facets. For publication-grade
        # evaluation, replace this weak mapping with clinician facet-level labels.
        for facet in facets:
            facet["gold_doc_ids"] = gold_docs
            facet["notes"] = "Auto-seeded from TREC CDS qrels; manually refine for final paper."
        rows.append({
            "case_id": qid,
            "dataset": dataset_name,
            "query": topic_text(query),
            "query_type": qtype,
            "patient_id": f"trec-cds-{qid}",
            "patient_context": patient_context,
            "gold_doc_ids": gold_docs,
            "gold_facets": facets,
            "constraints": {
                "clinical_scope": "clinician_cds",
                "source_policy": "ANY",
                "min_evidence_grade": 0.3,
            },
            "metadata": {
                "relevance_grades": relevance_by_qid.get(qid, {}),
                "summary": getattr(query, "summary", None),
                "description": getattr(query, "description", None),
                "source_dataset": dataset_name,
            },
        })
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} cases to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pmc/v2/trec-cds-2016")
    parser.add_argument("--out", default="data/trec_cds_2016/cases.jsonl")
    parser.add_argument("--max-topics", type=int, default=None)
    parser.add_argument("--max-docs-per-topic", type=int, default=200)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prepare(args.dataset, args.out, args.max_topics, args.max_docs_per_topic)


if __name__ == "__main__":
    main()
