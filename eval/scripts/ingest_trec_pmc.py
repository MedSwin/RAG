#!/usr/bin/env python3
"""Bulk-ingest a limited PMC corpus subset into MedSwin.

For full experiments, use the complete TREC CDS PMC collection or the judged
pool plus hard negatives. This script is intentionally batchable and resumable.

Example:
  python scripts/ingest_trec_pmc.py --dataset pmc/v2 --limit 1000 \
    --medswin-base-url http://localhost:8100 --org-id bench-org
"""
from __future__ import annotations

import argparse
from itertools import islice
from typing import Any

import httpx
import ir_datasets


def doc_to_payload(doc: Any) -> dict[str, Any]:
    title = getattr(doc, "title", "") or "Untitled PMC article"
    abstract = getattr(doc, "abstract", "") or ""
    body = getattr(doc, "body", "") or ""
    text = f"{title}\n\nAbstract\n{abstract}\n\nBody\n{body}".strip()
    return {
        "doc_id": str(getattr(doc, "doc_id")),
        "title": title,
        "version": "trec-cds-pmc-snapshot",
        "source_reliability": 0.75,
        "evidence_grade": {
            "label": "biomedical_literature",
            "score": 0.65,
            "source_reliability": 0.75,
        },
        "tags": ["TREC-CDS", "PMC", "biomedical-literature"],
        "metadata": {"dataset": "TREC CDS PMC"},
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pmc/v2")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--medswin-base-url", default="http://localhost:8100")
    parser.add_argument("--org-id", default="bench-org")
    args = parser.parse_args()

    dataset = ir_datasets.load(args.dataset)
    docs_iter = islice(dataset.docs_iter(), args.limit)
    batch = []
    total = 0
    with httpx.Client(timeout=180.0) as client:
        for doc in docs_iter:
            batch.append(doc_to_payload(doc))
            if len(batch) >= args.batch_size:
                resp = client.post(
                    f"{args.medswin_base_url.rstrip('/')}/api/v1/medswin/ingest",
                    params={"source_type": "LIT", "org_id": args.org_id},
                    json=batch,
                )
                resp.raise_for_status()
                total += len(batch)
                print(f"ingested {total}")
                batch = []
        if batch:
            resp = client.post(
                f"{args.medswin_base_url.rstrip('/')}/api/v1/medswin/ingest",
                params={"source_type": "LIT", "org_id": args.org_id},
                json=batch,
            )
            resp.raise_for_status()
            total += len(batch)
            print(f"ingested {total}")


if __name__ == "__main__":
    main()
