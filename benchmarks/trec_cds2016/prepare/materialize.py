#!/usr/bin/env python3
"""Materialize the MedSwin document layer for the complete TREC runtime.

The scalable full-corpus preparer writes embedded chunks directly so it can
batch cloud embeddings and build disk-backed indexes without issuing 1.25M API
requests. The normal MedSwin ingest contract, however, persists both Document
and Chunk records. This script restores that document metadata layer from the
canonical prepared chunks using server-side Mongo aggregation.

It creates exactly one LIT document for every distinct PMC doc_id and one EMR
document for every benchmark case note. No raw article body is duplicated in the
documents collection; the Document schema is metadata-only and chunk text
remains the retrieval source of truth.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings
from app.core.database import get_sync_database

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_LIT_DOCUMENTS = 1_255_260
EXPECTED_EMR_DOCUMENTS = 30
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _ensure_tenant_indexes(documents) -> None:
    """Synchronously migrate the historical global doc_id unique index."""
    info = documents.index_information()
    for name, spec in info.items():
        keys = spec.get("key") or []
        if keys == [("doc_id", 1)] and spec.get("unique"):
            documents.drop_index(name)
    documents.create_index([("org_id", 1), ("doc_id", 1)], unique=True)
    documents.create_index([("org_id", 1), ("source_type", 1)])
    documents.create_index([("org_id", 1), ("patient_id", 1)])


def _source_counts(documents, org_id: str) -> tuple[int, int, int]:
    lit = documents.count_documents({"org_id": org_id, "source_type": "LIT"})
    emr = documents.count_documents({"org_id": org_id, "source_type": "EMR"})
    total = documents.count_documents({"org_id": org_id})
    return int(lit), int(emr), int(total)


def _expected_chunk_sources(chunks, org_id: str) -> None:
    lit_docs = list(
        chunks.aggregate(
            [
                {"$match": {"org_id": org_id, "source_type": "LIT"}},
                {"$group": {"_id": "$doc_id"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    emr_docs = list(
        chunks.aggregate(
            [
                {
                    "$match": {
                        "org_id": org_id,
                        "source_type": "EMR",
                        "metadata.benchmark_patient_context": True,
                    }
                },
                {"$group": {"_id": "$doc_id"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    lit_count = int(lit_docs[0]["count"]) if lit_docs else 0
    emr_count = int(emr_docs[0]["count"]) if emr_docs else 0
    if lit_count != EXPECTED_LIT_DOCUMENTS:
        raise RuntimeError(
            f"Cannot materialize documents: chunk corpus has {lit_count:,}/{EXPECTED_LIT_DOCUMENTS:,} LIT documents"
        )
    if emr_count != EXPECTED_EMR_DOCUMENTS:
        raise RuntimeError(
            f"Cannot materialize documents: chunk corpus has {emr_count}/{EXPECTED_EMR_DOCUMENTS} benchmark EMR documents"
        )


def _materialize_lit(chunks, documents, org_id: str) -> None:
    pipeline = [
        {"$match": {"org_id": org_id, "source_type": "LIT"}},
        {
            "$group": {
                "_id": "$doc_id",
                "title": {"$first": "$metadata.title"},
                "source_reliability": {"$first": "$source_reliability"},
                "evidence_grade": {"$first": "$evidence_grade"},
                "dataset": {"$first": "$metadata.dataset"},
                "dataset_doc_ordinal": {"$min": "$metadata.dataset_doc_ordinal"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": "$_id",
                "source_type": {"$literal": "LIT"},
                "title": {"$ifNull": ["$title", "Untitled PMC article"]},
                "version": {"$literal": "trec-cds-pmc-v2-full"},
                "effective_date": {"$literal": None},
                "patient_id": {"$literal": None},
                "org_id": {"$literal": org_id},
                "tags": {"$literal": ["TREC-CDS", "PMC", "biomedical-literature"]},
                "source_reliability": {"$ifNull": ["$source_reliability", 0.75]},
                "evidence_grade": "$evidence_grade",
                "metadata": {
                    "dataset": {"$ifNull": ["$dataset", EXPECTED_DATASET]},
                    "dataset_doc_ordinal": "$dataset_doc_ordinal",
                    "full_trec_runtime": {"$literal": True},
                },
                "created_at": {"$literal": datetime.now(timezone.utc)},
            }
        },
        {
            "$merge": {
                "into": documents.name,
                "on": ["org_id", "doc_id"],
                "whenMatched": "replace",
                "whenNotMatched": "insert",
            }
        },
    ]
    # Exhaust the cursor so server-side $merge completion is observed before
    # counts are checked.
    list(chunks.aggregate(pipeline, allowDiskUse=True))


def _materialize_emr(chunks, documents, org_id: str) -> None:
    pipeline = [
        {
            "$match": {
                "org_id": org_id,
                "source_type": "EMR",
                "metadata.benchmark_patient_context": True,
            }
        },
        {
            "$group": {
                "_id": "$doc_id",
                "patient_id": {"$first": "$patient_id"},
                "case_id": {"$first": "$metadata.benchmark_case_id"},
                "source_reliability": {"$first": "$source_reliability"},
                "evidence_grade": {"$first": "$evidence_grade"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": "$_id",
                "source_type": {"$literal": "EMR"},
                "title": {"$concat": ["TREC CDS case note ", {"$toString": "$case_id"}]},
                "version": {"$literal": "benchmark"},
                "effective_date": {"$literal": None},
                "patient_id": "$patient_id",
                "org_id": {"$literal": org_id},
                "tags": {"$literal": ["benchmark", "trec-cds-2016", "patient-context"]},
                "source_reliability": {"$ifNull": ["$source_reliability", 0.80]},
                "evidence_grade": "$evidence_grade",
                "metadata": {
                    "benchmark_case_id": "$case_id",
                    "dataset": {"$literal": EXPECTED_DATASET},
                    "benchmark_patient_context": {"$literal": True},
                    "full_trec_runtime": {"$literal": True},
                },
                "created_at": {"$literal": datetime.now(timezone.utc)},
            }
        },
        {
            "$merge": {
                "into": documents.name,
                "on": ["org_id", "doc_id"],
                "whenMatched": "replace",
                "whenNotMatched": "insert",
            }
        },
    ]
    list(chunks.aggregate(pipeline, allowDiskUse=True))


def materialize(org_id: str, *, force: bool = False) -> dict[str, Any]:
    db = get_sync_database()
    chunks = db["chunks"]
    documents = db["documents"]
    _ensure_tenant_indexes(documents)
    _expected_chunk_sources(chunks, org_id)

    lit_before, emr_before, total_before = _source_counts(documents, org_id)
    expected_total = EXPECTED_LIT_DOCUMENTS + EXPECTED_EMR_DOCUMENTS
    if not force and (
        lit_before == EXPECTED_LIT_DOCUMENTS
        and emr_before == EXPECTED_EMR_DOCUMENTS
        and total_before == expected_total
    ):
        payload = {
            "complete": True,
            "reused": True,
            "org_id": org_id,
            "lit_documents": lit_before,
            "emr_documents": emr_before,
            "total_documents": total_before,
            "completed_at": _now(),
        }
        _write_json(MANIFEST_DIR / f"full-trec-documents-{org_id}.json", payload)
        return payload

    # Rebuild only this benchmark tenant's metadata layer. Chunks/embeddings and
    # retrieval indexes are not touched.
    documents.delete_many({"org_id": org_id})
    _materialize_lit(chunks, documents, org_id)
    _materialize_emr(chunks, documents, org_id)

    lit, emr, total = _source_counts(documents, org_id)
    if lit != EXPECTED_LIT_DOCUMENTS or emr != EXPECTED_EMR_DOCUMENTS or total != expected_total:
        raise RuntimeError(
            "Document materialization incomplete: "
            f"LIT={lit:,}/{EXPECTED_LIT_DOCUMENTS:,}, EMR={emr}/{EXPECTED_EMR_DOCUMENTS}, "
            f"total={total:,}/{expected_total:,}"
        )
    payload = {
        "complete": True,
        "reused": False,
        "org_id": org_id,
        "lit_documents": lit,
        "emr_documents": emr,
        "total_documents": total,
        "completed_at": _now(),
    }
    _write_json(MANIFEST_DIR / f"full-trec-documents-{org_id}.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default="bench-org")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = materialize(args.org_id, force=args.force)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
