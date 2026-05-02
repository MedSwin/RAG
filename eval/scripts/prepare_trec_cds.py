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
    --max-topics 30
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import ir_datasets

from app.io_utils import write_jsonl


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
    qtype = getattr(query, "type", None) or "clinical"
    return f"For this patient, what evidence is relevant to the clinical question type: {qtype}?"


def get_patient_context(query: Any) -> str:
    # TREC CDS query fields vary by track. In ir_datasets 2016, fields usually
    # include description and summary. If a note field exists, prefer it.
    for name in ("note", "description", "summary"):
        value = getattr(query, name, None)
        if value:
            return str(value)
    return ""


def prepare(dataset_name: str, out_path: str, max_topics: int | None) -> None:
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
        gold_docs = sorted(set(qrels_by_qid.get(qid, [])))
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
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prepare(args.dataset, args.out, args.max_topics)


if __name__ == "__main__":
    main()
