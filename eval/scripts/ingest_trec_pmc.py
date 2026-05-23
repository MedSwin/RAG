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
import json
import re
import random
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import ir_datasets


REFERENCE_TAIL_RE = re.compile(
    r"\n(?:references|bibliography|acknowledg(?:e)?ments?|author contributions?|appendix)\b",
    re.IGNORECASE,
)


def _token_count(text: str) -> int:
    return len(text.split())


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = text.split()
    return " ".join(tokens[:max_tokens]).strip()


def _strip_reference_tail(text: str) -> str:
    """Drop boilerplate-heavy tails that do not help retrieval."""
    if not text:
        return ""
    cleaned = REFERENCE_TAIL_RE.split(text, maxsplit=1)[0]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _body_excerpt(body: str, max_body_tokens: int) -> str:
    body = _strip_reference_tail(body)
    if not body:
        return ""
    # Root Cause vs Logic: PMC bodies often contain long reference tails and
    # boilerplate sections. The ingest payload keeps only the most evidence-rich
    # opening passage so embeddings stay within quota without discarding the
    # article's clinical signal.
    body = " ".join(body.split())
    return _truncate_tokens(body, max_body_tokens)


def _compose_text(title: str, abstract: str, body_excerpt: str, max_total_tokens: int) -> str:
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())
    body_excerpt = " ".join(body_excerpt.split())
    parts: list[str] = []
    budget = max(0, max_total_tokens)

    title_budget = min(32, budget)
    title_part = _truncate_tokens(title, title_budget)
    if title_part:
        parts.append(title_part)
    budget -= _token_count(title_part)

    if abstract and budget > 0:
        abstract_budget = min(max(64, max_total_tokens // 4), budget)
        abstract_part = _truncate_tokens(abstract, abstract_budget)
        if abstract_part:
            parts.extend(["Abstract", abstract_part])
        budget -= _token_count(abstract_part) + 1

    if body_excerpt and budget > 0:
        body_part = _truncate_tokens(body_excerpt, budget)
        if body_part:
            parts.extend(["Body excerpt", body_part])

    return "\n\n".join(part for part in parts if part).strip()


def doc_to_payload(doc: Any, max_body_tokens: int, max_total_tokens: int) -> dict[str, Any]:
    title = getattr(doc, "title", "") or "Untitled PMC article"
    abstract = getattr(doc, "abstract", "") or ""
    body = getattr(doc, "body", "") or ""
    body_excerpt = _body_excerpt(body, max_body_tokens)
    text = _compose_text(title, abstract, body_excerpt, max_total_tokens)
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


def _checkpoint_path(default_org_id: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("./data") / f"trec_pmc_{default_org_id}_checkpoint.json"


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    base_delay = min(60.0, 2.0 * (2 ** (attempt - 1)))
    if retry_after:
        try:
            hinted = max(0.0, float(retry_after))
        except ValueError:
            hinted = base_delay
        return max(base_delay, hinted) + random.uniform(0.0, 0.5)
    return base_delay + random.uniform(0.0, 0.5)


def post_batch(client: httpx.Client, base_url: str, org_id: str, batch: list[dict[str, Any]], max_attempts: int = 5) -> tuple[dict[str, Any], int]:
    retries = 0
    for attempt in range(1, max_attempts + 1):
        resp = client.post(
            f"{base_url.rstrip('/')}/api/v1/medswin/ingest",
            params={"source_type": "LIT", "org_id": org_id},
            json=batch,
        )
        if resp.status_code in {429, 500, 502, 503, 504}:
            retries += 1
            if attempt == max_attempts:
                resp.raise_for_status()
            time.sleep(_retry_delay(attempt, resp.headers.get("Retry-After")))
            continue
        resp.raise_for_status()
        return resp.json(), retries
    raise RuntimeError("Unreachable retry loop exit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pmc/v2")
    parser.add_argument("--limit", type=int, default=None, help="Ingest first N docs for smoke tests.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-body-tokens",
        "--max-body-chars",
        dest="max_body_tokens",
        type=int,
        default=1200,
        help="Maximum body excerpt token budget used for each PMC payload.",
    )
    parser.add_argument(
        "--request-delay-s",
        type=float,
        default=5.0,
        help="Delay between ingest POST batches in seconds to stay under cloud embedding quotas.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="HTTP timeout for ingest/reset/index requests in seconds.",
    )
    parser.add_argument("--medswin-base-url", default="http://localhost:8100")
    parser.add_argument("--org-id", default="bench-org")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--cloud-mode", action="store_true")
    parser.add_argument("--reset-org", action="store_true")
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    dataset = ir_datasets.load(args.dataset)
    docs_iter = iter_docs(dataset, args.limit, args.sample_size, args.seed)
    base_url = args.medswin_base_url.rstrip("/")
    batch: list[dict[str, Any]] = []
    total = 0
    total_retries = 0
    completed_doc_ids: set[str] = set()
    checkpoint_file = _checkpoint_path(args.org_id, args.checkpoint_path)
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        completed_doc_ids = set(checkpoint.get("completed_doc_ids") or [])
        total = int(checkpoint.get("completed_count") or 0)

    # Root Cause vs Logic: the old 180s client timeout was shorter than some
    # corpus ingest/index requests, so long-running benchmark loads could fail
    # mid-stream even when the runtime was healthy. The logic here makes the
    # timeout explicit and tunable so the ingest can finish deterministically.
    with httpx.Client(timeout=args.timeout_s) as client:
        current_batch_size = min(args.batch_size, 8) if args.cloud_mode else max(1, args.batch_size)
        current_delay_s = max(args.request_delay_s, 10.0) if args.cloud_mode else max(0.0, args.request_delay_s)
        target_batch_size = max(1, args.batch_size)

        if args.reset_org:
            resp = client.post(
                f"{base_url}/api/v1/storage/benchmark/reset",
                json={"org_id": args.org_id, "remove_indexes": True},
            )
            resp.raise_for_status()
            print(resp.json(), flush=True)
            if checkpoint_file.exists() and args.resume:
                checkpoint_file.unlink()

        for doc in docs_iter:
            doc_id = str(getattr(doc, "doc_id"))
            if doc_id in completed_doc_ids:
                continue
            batch.append(doc_to_payload(doc, args.max_body_tokens, args.max_body_tokens + 256))
            if len(batch) >= current_batch_size:
                payload, retries = post_batch(client, base_url, args.org_id, batch)
                total_retries += retries
                total += len(batch)
                completed_doc_ids.update(item["doc_id"] for item in batch)
                print({"ingested": total, "batch_size": len(batch), "retries": retries, "response": payload}, flush=True)
                batch = []
                # Root Cause vs Logic: the benchmark corpus runner can emit
                # back-to-back ingest requests faster than the cloud embedding
                # deployment releases quota. A small delay between batches keeps
                # the corpus build from amplifying 429 retries into a long-lived
                # quota storm.
                if retries > 0:
                    current_batch_size = max(1, current_batch_size // 2)
                    current_delay_s = min(max(current_delay_s * 2.0, args.request_delay_s or 0.0), 120.0)
                else:
                    current_batch_size = min(target_batch_size, current_batch_size + 1)
                    current_delay_s = max(0.0, current_delay_s * 0.9)
                if current_delay_s > 0:
                    time.sleep(current_delay_s)
                save_checkpoint(
                    checkpoint_file,
                    {
                        "dataset": args.dataset,
                        "org_id": args.org_id,
                        "completed_doc_ids": sorted(completed_doc_ids),
                        "completed_count": total,
                        "batch_size": current_batch_size,
                        "delay_s": current_delay_s,
                        "retries": total_retries,
                        "updated_at": time.time(),
                    },
                )
        if batch:
            payload, retries = post_batch(client, base_url, args.org_id, batch)
            total_retries += retries
            total += len(batch)
            completed_doc_ids.update(item["doc_id"] for item in batch)
            print({"ingested": total, "batch_size": len(batch), "retries": retries, "response": payload}, flush=True)
            save_checkpoint(
                checkpoint_file,
                {
                    "dataset": args.dataset,
                    "org_id": args.org_id,
                    "completed_doc_ids": sorted(completed_doc_ids),
                    "completed_count": total,
                    "batch_size": current_batch_size,
                    "delay_s": current_delay_s,
                    "retries": total_retries,
                    "updated_at": time.time(),
                },
            )

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
                json={"force_rebuild": True, "org_id": args.org_id},
                timeout=max(1800.0, args.timeout_s),
            )
            resp.raise_for_status()
            print(resp.json(), flush=True)

        save_checkpoint(
            checkpoint_file,
            {
                "dataset": args.dataset,
                "org_id": args.org_id,
                "completed_doc_ids": sorted(completed_doc_ids),
                "completed_count": total,
                "batch_size": current_batch_size,
                "delay_s": current_delay_s,
                "retries": total_retries,
                "updated_at": time.time(),
                "finished": True,
            },
        )


if __name__ == "__main__":
    main()
