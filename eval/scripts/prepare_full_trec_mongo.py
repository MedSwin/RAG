#!/usr/bin/env python3
"""Prepare Mongo persistence indexes before complete-TREC ingestion.

This is a lightweight schema migration that runs *before* millions of chunk
upserts. It removes historical global chunk/document uniqueness and redundant
Mongo text indexes, then establishes the tenant-scoped natural keys used by the
current repositories. Full lexical retrieval is provided by SQLite FTS5 BM25,
so a second Mongo full-body text index would only duplicate storage/write cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.database import get_sync_database


def _drop_chunk_legacy_indexes(chunks) -> list[str]:
    removed: list[str] = []
    for name, spec in chunks.index_information().items():
        keys = spec.get("key") or []
        if name == "_id_":
            continue
        if keys == [("chunk_id", 1)] and spec.get("unique"):
            chunks.drop_index(name)
            removed.append(name)
            continue
        if any(str(kind) == "text" for _field, kind in keys):
            chunks.drop_index(name)
            removed.append(name)
    return removed


def _drop_document_legacy_indexes(documents) -> list[str]:
    removed: list[str] = []
    for name, spec in documents.index_information().items():
        keys = spec.get("key") or []
        if name == "_id_":
            continue
        if keys == [("doc_id", 1)] and spec.get("unique"):
            documents.drop_index(name)
            removed.append(name)
    return removed


def prepare(org_id: str) -> dict[str, Any]:
    db = get_sync_database()
    chunks = db["chunks"]
    documents = db["documents"]

    removed_chunks = _drop_chunk_legacy_indexes(chunks)
    removed_documents = _drop_document_legacy_indexes(documents)

    chunks.create_index([("org_id", 1), ("chunk_id", 1)], unique=True)
    chunks.create_index([("org_id", 1), ("source_type", 1)])
    chunks.create_index([("org_id", 1), ("patient_id", 1)])
    chunks.create_index([("org_id", 1), ("doc_id", 1)])

    documents.create_index([("org_id", 1), ("doc_id", 1)], unique=True)
    documents.create_index([("org_id", 1), ("source_type", 1)])
    documents.create_index([("org_id", 1), ("patient_id", 1)])

    chunk_info = chunks.index_information()
    document_info = documents.index_information()
    chunk_text_indexes = [
        name
        for name, spec in chunk_info.items()
        if any(str(kind) == "text" for _field, kind in (spec.get("key") or []))
    ]
    if chunk_text_indexes:
        raise RuntimeError(f"Redundant Mongo chunk text indexes remain: {chunk_text_indexes}")

    chunk_compound_ok = any(
        spec.get("unique") and (spec.get("key") or []) == [("org_id", 1), ("chunk_id", 1)]
        for spec in chunk_info.values()
    )
    document_compound_ok = any(
        spec.get("unique") and (spec.get("key") or []) == [("org_id", 1), ("doc_id", 1)]
        for spec in document_info.values()
    )
    if not chunk_compound_ok or not document_compound_ok:
        raise RuntimeError("Tenant-scoped unique Mongo natural keys were not created")

    return {
        "strict_pass": True,
        "org_id": org_id,
        "removed_chunk_indexes": removed_chunks,
        "removed_document_indexes": removed_documents,
        "chunk_compound_unique": chunk_compound_ok,
        "document_compound_unique": document_compound_ok,
        "chunk_text_indexes": chunk_text_indexes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default="bench-org")
    args = parser.parse_args()
    print(json.dumps(prepare(args.org_id), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
