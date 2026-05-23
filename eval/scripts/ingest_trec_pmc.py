#!/usr/bin/env python3
"""Bulk-ingest a deterministic PMC corpus subset into MedSwin.

For full experiments, use the complete TREC CDS PMC collection or the judged
pool plus hard negatives. This script is intentionally batchable and resumable.

Example:
  python3 scripts/ingest_trec_pmc.py --dataset pmc/v2 --sample-size 5000 \
    --seed 1337 --reset-org --refresh-embeddings --build-index \
    --medswin-base-url http://localhost:8100 --org-id bench-org
"""
from __future__ import annotations

import argparse
import random
from typing import Any, Iterator

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


def iter_docs(dataset: Any, limit: int | None, sample_size: int, seed: int) -> Iterator[Any]:
    """Yield either first-N docs or a deterministic random sample."""
    if limit is not None:
        for idx, doc in enumerate(dataset.docs_iter()):
            if idx >= limit:
                break
            yield doc
        return

    # Motivation vs Logic: fixed-size benchmark corpora should be reproducible
    # without depending on upstream document ordering alone. Reservoir sampling
    # gives a deterministic random 5000-doc subset while streaming large PMC.
    rng = random.Random(seed)
    reservoir: list[Any] = []
    for idx, doc in enumerate(dataset.docs_iter()):
        if idx < sample_size:
            reservoir.append(doc)
            continue
        replacement = rng.randint(0, idx)
        if replacement < sample_size:
            reservoir[replacement] = doc
    for doc in reservoir:
        yield doc


def post_batch(client: httpx.Client, base_url: str, org_id: str, batch: list[dict[str, Any]]) -> None:
    resp = client.post(
        f"{base_url.rstrip('/')}/api/v1/medswin/ingest",
        params={"source_type": "LIT", "org_id": org_id},
        json=batch,
    )
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pmc/v2")
    parser.add_argument("--limit", type=int, default=None, help="Ingest first N docs for smoke tests.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="HTTP timeout for ingest/reset/index requests in seconds.",
    )
    parser.add_argument("--medswin-base-url", default="http://localhost:8100")
    parser.add_argument("--org-id", default="bench-org")
    parser.add_argument("--reset-org", action="store_true")
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()

    dataset = ir_datasets.load(args.dataset)
    docs_iter = iter_docs(dataset, args.limit, args.sample_size, args.seed)
    base_url = args.medswin_base_url.rstrip("/")
    batch: list[dict[str, Any]] = []
    total = 0

    # Root Cause vs Logic: the old 180s client timeout was shorter than some
    # corpus ingest/index requests, so long-running benchmark loads could fail
    # mid-stream even when the runtime was healthy. The logic here makes the
    # timeout explicit and tunable so the ingest can finish deterministically.
    with httpx.Client(timeout=args.timeout_s) as client:
        if args.reset_org:
            resp = client.post(
                f"{base_url}/api/v1/storage/benchmark/reset",
                json={"org_id": args.org_id, "remove_indexes": True},
            )
            resp.raise_for_status()
            print(resp.json(), flush=True)

        for doc in docs_iter:
            batch.append(doc_to_payload(doc))
            if len(batch) >= args.batch_size:
                post_batch(client, base_url, args.org_id, batch)
                total += len(batch)
                print(f"ingested {total}", flush=True)
                batch = []
        if batch:
            post_batch(client, base_url, args.org_id, batch)
            total += len(batch)
            print(f"ingested {total}", flush=True)

        if args.refresh_embeddings:
            resp = client.post(
                f"{base_url}/api/v1/storage/embeddings/refresh",
                json={"org_id": args.org_id},
                timeout=max(3600.0, args.timeout_s),
            )
            resp.raise_for_status()
            print(resp.json(), flush=True)

        if args.build_index:
            resp = client.post(
                f"{base_url}/api/v1/storage/index/build",
                json={"force_rebuild": True},
                timeout=max(1800.0, args.timeout_s),
            )
            resp.raise_for_status()
            print(resp.json(), flush=True)


if __name__ == "__main__":
    main()
