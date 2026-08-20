#!/usr/bin/env python3
"""Prepare Mongo persistence and validate complete-TREC resume state.

This lightweight preflight runs *before* millions of chunk upserts. It removes
historical global chunk/document uniqueness and redundant Mongo text indexes,
then establishes tenant-scoped natural keys.

It also makes the expensive publication build crash-safe. A compatible partial
checkpoint is accepted only when its entire persisted document-ordinal prefix
exists under the current preparation contract. Mongo may be slightly ahead of
the checkpoint after a crash because deterministic upserts can be replayed. If
the final literature flush succeeded but the terminal completion flag was not
sealed, this preflight verifies the full LIT corpus and repairs only that flag;
the normal preparer then re-materializes the 30 EMR notes and rebuilds indexes.

An incompatible checkpoint is reported but not mutated here. The corpus builder
remains authoritative: a normal resume fails closed on the mismatch, while an
operator-requested ``--reset-full-corpus`` can still discard the old state.
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

from app.core.config import settings
from app.core.database import get_sync_database
from eval.app.full_contract import (
    EXPECTED_CHUNKING_CONTRACT,
    EXPECTED_DATASET,
    EXPECTED_DOCUMENTS,
    PREPARATION_CONTRACT_VERSION,
    chunker_sha256,
)

MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"


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


def _distinct_count(collection, match: dict[str, Any], field: str) -> int:
    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": f"${field}"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    return int(rows[0]["count"]) if rows else 0


def _checkpoint_path(org_id: str) -> Path:
    return MANIFEST_DIR / f"full-trec-{org_id}-checkpoint.json"


def _expected_checkpoint_contract(org_id: str) -> dict[str, Any]:
    return {
        "dataset": EXPECTED_DATASET,
        "org_id": org_id,
        "expected_documents": EXPECTED_DOCUMENTS,
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_input_type": "document",
        "embedding_dim": settings.active_embedding_dimension(),
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
        "chunking_contract": EXPECTED_CHUNKING_CONTRACT,
        "chunker_sha256": chunker_sha256(),
        "preparation_contract_version": PREPARATION_CONTRACT_VERSION,
    }


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(checkpoint, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _validate_resume_checkpoint(chunks, org_id: str) -> dict[str, Any]:
    path = _checkpoint_path(org_id)
    if not path.exists():
        return {
            "checkpoint_present": False,
            "contract_mismatches": [],
            "next_doc_ordinal": 0,
            "prefix_documents": 0,
            "prefix_ordinals": 0,
            "ahead_documents": 0,
            "terminal_repaired": False,
        }

    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = _expected_checkpoint_contract(org_id)
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        # Do not mutate or inspect the incompatible corpus here. A normal run
        # will fail closed in prepare_full_trec_runtime._validate_checkpoint;
        # an explicit --reset-full-corpus is allowed to delete it there.
        return {
            "checkpoint_present": True,
            "contract_mismatches": mismatches,
            "next_doc_ordinal": int(checkpoint.get("next_doc_ordinal") or 0),
            "prefix_documents": 0,
            "prefix_ordinals": 0,
            "ahead_documents": 0,
            "terminal_repaired": False,
        }

    next_ordinal = int(checkpoint.get("next_doc_ordinal") or 0)
    if next_ordinal < 0 or next_ordinal > EXPECTED_DOCUMENTS:
        raise RuntimeError(f"Invalid full-TREC checkpoint ordinal {next_ordinal}")
    completed = bool(checkpoint.get("completed"))
    if completed and next_ordinal != EXPECTED_DOCUMENTS:
        raise RuntimeError("Checkpoint claims completion before the terminal document ordinal")

    if next_ordinal == 0:
        return {
            "checkpoint_present": True,
            "contract_mismatches": [],
            "next_doc_ordinal": 0,
            "prefix_documents": 0,
            "prefix_ordinals": 0,
            "ahead_documents": 0,
            "terminal_repaired": False,
        }

    current_sha = chunker_sha256()
    wrong_contract = chunks.count_documents(
        {
            "org_id": org_id,
            "source_type": "LIT",
            "$or": [
                {"metadata.full_trec_runtime": {"$ne": True}},
                {"metadata.preparation_contract_version": {"$ne": PREPARATION_CONTRACT_VERSION}},
                {"metadata.chunker_sha256": {"$ne": current_sha}},
            ],
        }
    )
    if wrong_contract:
        raise RuntimeError(
            f"Cannot resume: {wrong_contract:,} persisted literature chunks use a different preparation contract"
        )

    invalid_ordinals = chunks.count_documents(
        {
            "org_id": org_id,
            "source_type": "LIT",
            "$or": [
                {"metadata.dataset_doc_ordinal": {"$exists": False}},
                {"metadata.dataset_doc_ordinal": {"$lt": 0}},
                {"metadata.dataset_doc_ordinal": {"$gte": EXPECTED_DOCUMENTS}},
            ],
        }
    )
    if invalid_ordinals:
        raise RuntimeError(f"Cannot resume: {invalid_ordinals:,} literature chunks have invalid dataset ordinals")

    prefix_match = {
        "org_id": org_id,
        "source_type": "LIT",
        "metadata.dataset_doc_ordinal": {"$gte": 0, "$lt": next_ordinal},
    }
    prefix_documents = _distinct_count(chunks, prefix_match, "doc_id")
    prefix_ordinals = _distinct_count(chunks, prefix_match, "metadata.dataset_doc_ordinal")
    if prefix_documents != next_ordinal or prefix_ordinals != next_ordinal:
        raise RuntimeError(
            "Checkpointed full-TREC prefix is incomplete: "
            f"documents={prefix_documents:,}/{next_ordinal:,}, "
            f"ordinals={prefix_ordinals:,}/{next_ordinal:,}. "
            "Reset the benchmark corpus rather than skipping missing source documents."
        )

    ahead_documents = _distinct_count(
        chunks,
        {
            "org_id": org_id,
            "source_type": "LIT",
            "metadata.dataset_doc_ordinal": {"$gte": next_ordinal},
        },
        "doc_id",
    )

    terminal_repaired = False
    if next_ordinal == EXPECTED_DOCUMENTS and not completed:
        active_filter = {
            "org_id": org_id,
            "source_type": "LIT",
            "embedding_space": settings.active_embedding_space(),
            "embedding_model": settings.active_embedding_model(),
            "embedding_dim": settings.active_embedding_dimension(),
            "embedding": {"$exists": True, "$type": "array", "$ne": []},
        }
        active_chunks = chunks.count_documents(active_filter)
        total_lit_chunks = chunks.count_documents({"org_id": org_id, "source_type": "LIT"})
        if total_lit_chunks <= 0 or active_chunks != total_lit_chunks:
            raise RuntimeError(
                "Terminal checkpoint cannot be repaired because not every LIT chunk is in the active embedding space"
            )
        checkpoint["completed"] = True
        _write_checkpoint(path, checkpoint)
        terminal_repaired = True

    return {
        "checkpoint_present": True,
        "contract_mismatches": [],
        "next_doc_ordinal": next_ordinal,
        "prefix_documents": prefix_documents,
        "prefix_ordinals": prefix_ordinals,
        "ahead_documents": ahead_documents,
        "terminal_repaired": terminal_repaired,
    }


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
    chunks.create_index([("org_id", 1), ("metadata.dataset_doc_ordinal", 1)])

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

    resume = _validate_resume_checkpoint(chunks, org_id)

    return {
        "strict_pass": True,
        "org_id": org_id,
        "removed_chunk_indexes": removed_chunks,
        "removed_document_indexes": removed_documents,
        "chunk_compound_unique": chunk_compound_ok,
        "document_compound_unique": document_compound_ok,
        "chunk_text_indexes": chunk_text_indexes,
        "resume": resume,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default="bench-org")
    args = parser.parse_args()
    print(json.dumps(prepare(args.org_id), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())