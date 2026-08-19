#!/usr/bin/env python3
"""Fail-closed integrity verification for the complete TREC-CDS runtime.

This verifier is intentionally independent of the ingestion checkpoint. It
proves that the *current persisted runtime* still represents all 1,255,260 PMC
collection documents and that every persisted chunk is present in the active
Cohere Embed v4 vector space, SQLite BM25 index, and HNSW index. It also verifies
all 30 benchmark patient contexts and every positive-qrel document referenced by
the full case file.

A successful source-data iteration is not sufficient: deletion/corruption after
a checkpoint must not be allowed to masquerade as a publication-complete run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hnswlib

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
for root in (REPO_ROOT, EVAL_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.config import settings
from app.core.database import get_sync_database

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_QRELS = 37_707
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _cases(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    gold: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            gold.update(str(value) for value in row.get("gold_doc_ids") or [] if str(value).strip())
    if len(rows) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} benchmark cases, found {len(rows)}")
    case_ids = {str(row.get("case_id")) for row in rows}
    if len(case_ids) != EXPECTED_QUERIES:
        raise RuntimeError("Benchmark case IDs are not unique")
    if not gold:
        raise RuntimeError("Full benchmark case file contains no positive qrel documents")
    return rows, gold


def _distinct_literature_docs(coll: Any, org_id: str) -> int:
    # Server-side grouping avoids transferring or materializing 1.25M IDs in
    # the verification process. allowDiskUse keeps this deterministic on large
    # Mongo installations with constrained aggregation memory.
    rows = list(
        coll.aggregate(
            [
                {"$match": {"org_id": org_id, "source_type": "LIT"}},
                {"$group": {"_id": "$doc_id"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    return int(rows[0]["count"]) if rows else 0


def _distinct_case_patients(coll: Any, org_id: str) -> int:
    rows = list(
        coll.aggregate(
            [
                {
                    "$match": {
                        "org_id": org_id,
                        "source_type": "EMR",
                        "metadata.benchmark_patient_context": True,
                    }
                },
                {"$group": {"_id": "$patient_id"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    return int(rows[0]["count"]) if rows else 0


def _fts_count(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(f"BM25 FTS artifact is missing: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        count = int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        # Exercise the ranking function itself so a malformed/non-FTS database
        # cannot pass merely because it has a table with the expected row count.
        conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
            ('"clinical"',),
        ).fetchall()
        return count
    finally:
        conn.close()


def _mapping_count(path: Path) -> int:
    if not path.exists():
        raise RuntimeError(f"HNSW mapping artifact is missing: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT count(*) FROM mapping").fetchone()[0])
    finally:
        conn.close()


def _hnsw_count(path: Path, dimension: int) -> int:
    if not path.exists():
        raise RuntimeError(f"HNSW index artifact is missing: {path}")
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(path))
    return int(index.get_current_count())


def verify(org_id: str, cases_path: Path) -> dict[str, Any]:
    runtime_path = MANIFEST_DIR / f"full-trec-runtime-{org_id}.json"
    runtime = _load_json(runtime_path)
    rows, gold_doc_ids = _cases(cases_path)

    if runtime.get("complete") is not True:
        raise RuntimeError("Full runtime manifest is not marked complete")
    if runtime.get("dataset") != EXPECTED_DATASET:
        raise RuntimeError(f"Unexpected dataset: {runtime.get('dataset')}")
    if int(runtime.get("expected_documents") or 0) != EXPECTED_DOCS:
        raise RuntimeError("Runtime manifest does not target the complete TREC collection")
    if int(runtime.get("queries") or 0) != EXPECTED_QUERIES:
        raise RuntimeError("Runtime manifest does not contain all 30 TREC topics")
    if int(runtime.get("qrels") or 0) != EXPECTED_QRELS:
        raise RuntimeError("Runtime manifest qrel count is incomplete")
    if runtime.get("embedding_model") != "embed-v-4-0":
        raise RuntimeError(f"Unexpected embedding model: {runtime.get('embedding_model')}")
    if runtime.get("embedding_input_type") != "document":
        raise RuntimeError("Corpus manifest does not prove document-mode embedding")

    db = get_sync_database()
    coll = db["chunks"]
    base = {"org_id": org_id}
    total_chunks = coll.count_documents(base)
    lit_chunks = coll.count_documents({**base, "source_type": "LIT"})
    emr_chunks = coll.count_documents({**base, "source_type": "EMR"})
    literature_docs = _distinct_literature_docs(coll, org_id)
    case_patients = _distinct_case_patients(coll, org_id)

    active_filter = {
        **base,
        "embedding_space": settings.active_embedding_space(),
        "embedding_model": settings.active_embedding_model(),
        "embedding_dim": settings.active_embedding_dimension(),
        "embedding": {"$exists": True, "$type": "array", "$ne": []},
    }
    active_chunks = coll.count_documents(active_filter)
    if total_chunks <= 0:
        raise RuntimeError("Benchmark corpus is empty")
    if literature_docs != EXPECTED_DOCS:
        raise RuntimeError(
            f"Persisted TREC literature coverage is {literature_docs:,}/{EXPECTED_DOCS:,} distinct documents"
        )
    if case_patients != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Persisted benchmark EMR coverage is {case_patients}/{EXPECTED_QUERIES} patient cases"
        )
    if active_chunks != total_chunks:
        raise RuntimeError(
            f"Only {active_chunks:,}/{total_chunks:,} chunks are in the active embedding space"
        )

    gold_present = set(
        str(value)
        for value in coll.distinct(
            "doc_id",
            {"org_id": org_id, "source_type": "LIT", "doc_id": {"$in": list(gold_doc_ids)}},
        )
    )
    if gold_present != gold_doc_ids:
        missing = sorted(gold_doc_ids - gold_present)
        raise RuntimeError(f"Missing {len(missing)} positive-qrel documents; sample={missing[:20]}")

    bm25_info = runtime.get("bm25") or {}
    hnsw_info = runtime.get("hnsw") or {}
    fts_path = Path(str(bm25_info.get("path") or ""))
    index_path = Path(str(hnsw_info.get("index_path") or settings.HNSW_INDEX_PATH))
    mapping_path = Path(str(hnsw_info.get("mapping_path") or settings.HNSW_MAPPING_PATH))
    fts_rows = _fts_count(fts_path)
    mapping_rows = _mapping_count(mapping_path)
    hnsw_vectors = _hnsw_count(index_path, settings.active_embedding_dimension())
    if fts_rows != total_chunks:
        raise RuntimeError(f"BM25 FTS rows={fts_rows:,}, corpus chunks={total_chunks:,}")
    if mapping_rows != total_chunks:
        raise RuntimeError(f"HNSW mapping rows={mapping_rows:,}, corpus chunks={total_chunks:,}")
    if hnsw_vectors != total_chunks:
        raise RuntimeError(f"HNSW vectors={hnsw_vectors:,}, corpus chunks={total_chunks:,}")

    verification = {
        "strict_pass": True,
        "verified_at": _now(),
        "dataset": EXPECTED_DATASET,
        "org_id": org_id,
        "expected_documents": EXPECTED_DOCS,
        "persisted_literature_documents": literature_docs,
        "queries": len(rows),
        "qrels": EXPECTED_QRELS,
        "positive_qrel_documents": len(gold_doc_ids),
        "positive_qrel_documents_present": len(gold_present),
        "case_patients": case_patients,
        "total_chunks": total_chunks,
        "lit_chunks": lit_chunks,
        "emr_chunks": emr_chunks,
        "active_embedding_chunks": active_chunks,
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_dimension": settings.active_embedding_dimension(),
        "corpus_embedding_input_type": "document",
        "bm25_rows": fts_rows,
        "hnsw_vectors": hnsw_vectors,
        "hnsw_mapping_rows": mapping_rows,
        "cases_path": str(cases_path.resolve()),
        "runtime_manifest": str(runtime_path.resolve()),
    }
    output = MANIFEST_DIR / f"full-trec-verification-{org_id}.json"
    _write_json(output, verification)
    verification["verification_path"] = str(output.resolve())
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", default="bench-org")
    parser.add_argument("--cases-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = _load_json(MANIFEST_DIR / f"full-trec-runtime-{args.org_id}.json")
    cases_path = Path(args.cases_path or runtime.get("cases_path") or "")
    if not str(cases_path):
        raise RuntimeError("No full TREC cases path is configured")
    result = verify(args.org_id, cases_path)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
