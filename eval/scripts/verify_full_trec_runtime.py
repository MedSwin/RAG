#!/usr/bin/env python3
"""Fail-closed integrity verification for the complete TREC-CDS runtime.

This verifier is intentionally independent of the ingestion checkpoint. It
proves that the *current persisted runtime* still represents all 1,255,260 PMC
collection documents and all 30 benchmark EMR cases; that the MedSwin Document
and Chunk layers agree; and that every persisted chunk is represented in the
active Cohere Embed v4 vector space, SQLite BM25 index, and HNSW label mapping.

Counts alone are insufficient: equal-size stores can contain different IDs. We
therefore compute order-independent cryptographic multiset fingerprints over
cross-store identifiers and require exact equality. In addition, deterministic
HNSW labels are sampled and their stored vectors are compared with the Mongo
embedding for the mapped chunk ID, catching vector/label mis-association that
count and ID-set checks alone cannot detect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import hnswlib
import numpy as np

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
EXPECTED_EMR_DOCS = 30
EXPECTED_EMBEDDING = "embed-v-4-0"
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"
_FINGERPRINT_MASK = (1 << 256) - 1


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


def _distinct_count(coll: Any, match: dict[str, Any], field: str) -> int:
    rows = list(
        coll.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": f"${field}"}},
                {"$count": "count"},
            ],
            allowDiskUse=True,
        )
    )
    return int(rows[0]["count"]) if rows else 0


def _fingerprint_ids(values: Iterable[str]) -> dict[str, Any]:
    """Return an order-independent SHA-256 multiset fingerprint."""
    count = 0
    xor_value = 0
    sum_value = 0
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            raise RuntimeError("Encountered an empty identifier while fingerprinting full runtime")
        digest = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")
        xor_value ^= digest
        sum_value = (sum_value + digest) & _FINGERPRINT_MASK
        count += 1
    return {
        "count": count,
        "xor_sha256": f"{xor_value:064x}",
        "sum_sha256_mod_2_256": f"{sum_value:064x}",
    }


def _mongo_chunk_ids(coll: Any, org_id: str, batch_size: int = 4096) -> Iterator[str]:
    cursor = coll.find({"org_id": org_id}, {"_id": 0, "chunk_id": 1}, batch_size=batch_size)
    for row in cursor:
        yield str(row.get("chunk_id") or "")


def _distinct_chunk_doc_ids(coll: Any, org_id: str, source_type: str) -> Iterator[str]:
    cursor = coll.aggregate(
        [
            {"$match": {"org_id": org_id, "source_type": source_type}},
            {"$group": {"_id": "$doc_id"}},
        ],
        allowDiskUse=True,
    )
    for row in cursor:
        yield str(row.get("_id") or "")


def _document_doc_ids(documents: Any, org_id: str, source_type: str, batch_size: int = 4096) -> Iterator[str]:
    cursor = documents.find(
        {"org_id": org_id, "source_type": source_type},
        {"_id": 0, "doc_id": 1},
        batch_size=batch_size,
    )
    for row in cursor:
        yield str(row.get("doc_id") or "")


def _sqlite_chunk_ids(conn: sqlite3.Connection, table: str, batch_size: int = 8192) -> Iterator[str]:
    cursor = conn.execute(f"SELECT chunk_id FROM {table}")
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield str(row[0] or "")


def _fts_integrity(path: Path, org_id: str) -> tuple[int, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"BM25 FTS artifact is missing: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        count = int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        org_rows = int(conn.execute("SELECT count(*) FROM chunks_fts WHERE org_id = ?", (org_id,)).fetchone()[0])
        if org_rows != count:
            raise RuntimeError(f"BM25 FTS contains {count - org_rows:,} rows outside benchmark org {org_id}")
        conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
            ('"clinical"',),
        ).fetchall()
        fingerprint = _fingerprint_ids(_sqlite_chunk_ids(conn, "chunks_fts"))
        return count, fingerprint
    finally:
        conn.close()


def _mapping_integrity(path: Path) -> tuple[int, dict[str, Any], dict[str, int | None]]:
    if not path.exists():
        raise RuntimeError(f"HNSW mapping artifact is missing: {path}")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT count(*), min(label), max(label) FROM mapping").fetchone()
        count = int(row[0] or 0)
        label_bounds = {
            "min": int(row[1]) if row[1] is not None else None,
            "max": int(row[2]) if row[2] is not None else None,
        }
        if count > 0 and (label_bounds["min"] != 0 or label_bounds["max"] != count - 1):
            raise RuntimeError(
                f"HNSW mapping labels are not contiguous 0..{count - 1}: bounds={label_bounds}"
            )
        fingerprint = _fingerprint_ids(_sqlite_chunk_ids(conn, "mapping"))
        return count, fingerprint, label_bounds
    finally:
        conn.close()


def _hnsw_count(path: Path, dimension: int) -> int:
    if not path.exists():
        raise RuntimeError(f"HNSW index artifact is missing: {path}")
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(path))
    return int(index.get_current_count())


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _hnsw_vector_alignment(
    index_path: Path,
    mapping_path: Path,
    coll: Any,
    org_id: str,
    dimension: int,
    total_vectors: int,
) -> dict[str, Any]:
    """Deterministically sample HNSW label->vector->Mongo identity alignment."""
    if total_vectors <= 0:
        raise RuntimeError("Cannot verify HNSW vector alignment for an empty index")
    requested = max(1, int(os.getenv("FULL_VERIFY_HNSW_SAMPLES", "32")))
    sample_count = min(requested, total_vectors)
    if sample_count == 1:
        labels = [0]
    else:
        labels = sorted(
            set(int(value) for value in np.linspace(0, total_vectors - 1, num=sample_count, dtype=np.int64))
        )

    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(index_path))
    conn = sqlite3.connect(f"file:{mapping_path.resolve()}?mode=ro", uri=True)
    min_cosine = 1.0
    samples: list[dict[str, Any]] = []
    try:
        for label in labels:
            row = conn.execute("SELECT chunk_id FROM mapping WHERE label = ?", (label,)).fetchone()
            if row is None:
                raise RuntimeError(f"HNSW mapping has no chunk_id for sampled label {label}")
            chunk_id = str(row[0])
            mongo = coll.find_one(
                {"org_id": org_id, "chunk_id": chunk_id},
                {"_id": 0, "embedding": 1, "embedding_dim": 1, "embedding_space": 1},
            )
            if not mongo:
                raise RuntimeError(f"Mongo has no chunk for sampled HNSW label {label} -> {chunk_id}")
            embedding = mongo.get("embedding") or []
            if len(embedding) != dimension:
                raise RuntimeError(
                    f"Mongo embedding dimension for {chunk_id} is {len(embedding)}, expected {dimension}"
                )
            stored = np.asarray(index.get_items([label])[0], dtype=np.float32)
            similarity = _cosine(stored, np.asarray(embedding, dtype=np.float32))
            min_cosine = min(min_cosine, similarity)
            # hnswlib normalizes stored vectors for cosine space, so cosine
            # equivalence rather than element-wise equality is the correct check.
            if similarity < 0.9999:
                raise RuntimeError(
                    f"HNSW label/vector alignment mismatch for label={label} chunk_id={chunk_id}: cosine={similarity:.6f}"
                )
            samples.append({"label": label, "chunk_id": chunk_id, "cosine": similarity})
    finally:
        conn.close()

    return {
        "sample_count": len(samples),
        "requested_sample_count": requested,
        "minimum_cosine": min_cosine,
        "first_sample": samples[0] if samples else None,
        "last_sample": samples[-1] if samples else None,
    }


def verify(org_id: str, cases_path: Path) -> dict[str, Any]:
    runtime_path = MANIFEST_DIR / f"full-trec-runtime-{org_id}.json"
    runtime = _load_json(runtime_path)
    rows, gold_doc_ids = _cases(cases_path)

    if runtime.get("complete") is not True:
        raise RuntimeError("Full runtime manifest is not marked complete")
    if runtime.get("dataset") != EXPECTED_DATASET:
        raise RuntimeError(f"Unexpected dataset: {runtime.get('dataset')}")
    if runtime.get("org_id") != org_id:
        raise RuntimeError(f"Runtime manifest org {runtime.get('org_id')!r} != requested {org_id!r}")
    if int(runtime.get("expected_documents") or 0) != EXPECTED_DOCS:
        raise RuntimeError("Runtime manifest does not target the complete TREC collection")
    if int(runtime.get("queries") or 0) != EXPECTED_QUERIES:
        raise RuntimeError("Runtime manifest does not contain all 30 TREC topics")
    if int(runtime.get("qrels") or 0) != EXPECTED_QRELS:
        raise RuntimeError("Runtime manifest qrel count is incomplete")
    if runtime.get("embedding_model") != EXPECTED_EMBEDDING:
        raise RuntimeError(f"Unexpected embedding model: {runtime.get('embedding_model')}")
    if runtime.get("embedding_input_type") != "document":
        raise RuntimeError("Corpus manifest does not prove document-mode embedding")
    if runtime.get("embedding_space") != settings.active_embedding_space():
        raise RuntimeError(
            f"Runtime embedding space {runtime.get('embedding_space')!r} != active {settings.active_embedding_space()!r}"
        )
    if int(runtime.get("embedding_dim") or 0) != settings.active_embedding_dimension():
        raise RuntimeError("Runtime embedding dimension does not match the active evaluation dimension")
    if runtime.get("chunking") != "app.medswin.chunking.section_chunks/full-body":
        raise RuntimeError(f"Unexpected full-corpus chunking contract: {runtime.get('chunking')!r}")

    db = get_sync_database()
    coll = db["chunks"]
    documents = db["documents"]
    base = {"org_id": org_id}

    total_chunks = coll.count_documents(base)
    lit_chunks = coll.count_documents({**base, "source_type": "LIT"})
    emr_chunks = coll.count_documents({**base, "source_type": "EMR"})
    if total_chunks <= 0:
        raise RuntimeError("Benchmark corpus is empty")
    if total_chunks != lit_chunks + emr_chunks:
        raise RuntimeError(
            f"Benchmark chunk corpus contains unsupported extra source types: total={total_chunks:,}, "
            f"LIT+EMR={lit_chunks + emr_chunks:,}"
        )

    literature_docs = _distinct_count(coll, {**base, "source_type": "LIT"}, "doc_id")
    case_docs = _distinct_count(
        coll,
        {**base, "source_type": "EMR", "metadata.benchmark_patient_context": True},
        "doc_id",
    )
    case_patients = _distinct_count(
        coll,
        {**base, "source_type": "EMR", "metadata.benchmark_patient_context": True},
        "patient_id",
    )
    if literature_docs != EXPECTED_DOCS:
        raise RuntimeError(
            f"Persisted TREC literature coverage is {literature_docs:,}/{EXPECTED_DOCS:,} distinct documents"
        )
    if case_docs != EXPECTED_EMR_DOCS or case_patients != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Persisted benchmark EMR coverage is documents={case_docs}/{EXPECTED_EMR_DOCS}, "
            f"patients={case_patients}/{EXPECTED_QUERIES}"
        )

    active_filter = {
        **base,
        "embedding_space": settings.active_embedding_space(),
        "embedding_model": settings.active_embedding_model(),
        "embedding_dim": settings.active_embedding_dimension(),
        "embedding": {"$exists": True, "$type": "array", "$ne": []},
    }
    active_chunks = coll.count_documents(active_filter)
    if active_chunks != total_chunks:
        raise RuntimeError(f"Only {active_chunks:,}/{total_chunks:,} chunks are in the active embedding space")

    lit_document_records = documents.count_documents({**base, "source_type": "LIT"})
    emr_document_records = documents.count_documents({**base, "source_type": "EMR"})
    total_document_records = documents.count_documents(base)
    expected_document_records = EXPECTED_DOCS + EXPECTED_EMR_DOCS
    if (
        lit_document_records != EXPECTED_DOCS
        or emr_document_records != EXPECTED_EMR_DOCS
        or total_document_records != expected_document_records
    ):
        raise RuntimeError(
            "MedSwin Document layer incomplete or contaminated: "
            f"LIT={lit_document_records:,}/{EXPECTED_DOCS:,}, "
            f"EMR={emr_document_records}/{EXPECTED_EMR_DOCS}, "
            f"total={total_document_records:,}/{expected_document_records:,}"
        )
    bad_document_metadata = documents.count_documents(
        {
            "org_id": org_id,
            "$or": [
                {"doc_id": {"$exists": False}},
                {"doc_id": ""},
                {"title": {"$exists": False}},
                {"title": ""},
                {"evidence_grade": {"$exists": False}},
                {"source_reliability": {"$exists": False}},
                {"metadata.full_trec_runtime": {"$ne": True}},
            ],
        }
    )
    if bad_document_metadata:
        raise RuntimeError(f"{bad_document_metadata:,} benchmark Document records fail required metadata checks")

    lit_chunk_doc_fingerprint = _fingerprint_ids(_distinct_chunk_doc_ids(coll, org_id, "LIT"))
    lit_document_fingerprint = _fingerprint_ids(_document_doc_ids(documents, org_id, "LIT"))
    emr_chunk_doc_fingerprint = _fingerprint_ids(_distinct_chunk_doc_ids(coll, org_id, "EMR"))
    emr_document_fingerprint = _fingerprint_ids(_document_doc_ids(documents, org_id, "EMR"))
    if lit_chunk_doc_fingerprint != lit_document_fingerprint:
        raise RuntimeError("LIT Document doc_id fingerprint does not match the chunk-derived PMC document universe")
    if emr_chunk_doc_fingerprint != emr_document_fingerprint:
        raise RuntimeError("EMR Document doc_id fingerprint does not match the chunk-derived benchmark note universe")

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

    mongo_fingerprint = _fingerprint_ids(_mongo_chunk_ids(coll, org_id))
    if mongo_fingerprint["count"] != total_chunks:
        raise RuntimeError("Mongo chunk fingerprint count changed during verification")

    bm25_info = runtime.get("bm25") or {}
    hnsw_info = runtime.get("hnsw") or {}
    if bm25_info.get("backend") != "sqlite-fts5-bm25":
        raise RuntimeError(f"Unexpected BM25 backend: {bm25_info.get('backend')!r}")
    if hnsw_info.get("mapping_backend") != "sqlite":
        raise RuntimeError(f"Unexpected HNSW mapping backend: {hnsw_info.get('mapping_backend')!r}")
    if hnsw_info.get("embedding_model") != EXPECTED_EMBEDDING:
        raise RuntimeError("HNSW manifest embedding model does not match full runtime")
    if int(hnsw_info.get("embedding_dim") or 0) != settings.active_embedding_dimension():
        raise RuntimeError("HNSW manifest embedding dimension does not match full runtime")

    fts_path = Path(str(bm25_info.get("path") or ""))
    index_path = Path(str(hnsw_info.get("index_path") or settings.HNSW_INDEX_PATH))
    mapping_path = Path(str(hnsw_info.get("mapping_path") or settings.HNSW_MAPPING_PATH))
    fts_rows, fts_fingerprint = _fts_integrity(fts_path, org_id)
    mapping_rows, mapping_fingerprint, label_bounds = _mapping_integrity(mapping_path)
    hnsw_vectors = _hnsw_count(index_path, settings.active_embedding_dimension())

    if fts_rows != total_chunks:
        raise RuntimeError(f"BM25 FTS rows={fts_rows:,}, corpus chunks={total_chunks:,}")
    if mapping_rows != total_chunks:
        raise RuntimeError(f"HNSW mapping rows={mapping_rows:,}, corpus chunks={total_chunks:,}")
    if hnsw_vectors != total_chunks:
        raise RuntimeError(f"HNSW vectors={hnsw_vectors:,}, corpus chunks={total_chunks:,}")
    if fts_fingerprint != mongo_fingerprint:
        raise RuntimeError("BM25 FTS chunk-ID fingerprint does not match Mongo benchmark corpus")
    if mapping_fingerprint != mongo_fingerprint:
        raise RuntimeError("HNSW mapping chunk-ID fingerprint does not match Mongo benchmark corpus")

    hnsw_alignment = _hnsw_vector_alignment(
        index_path,
        mapping_path,
        coll,
        org_id,
        settings.active_embedding_dimension(),
        hnsw_vectors,
    )

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
        "case_documents": case_docs,
        "case_patients": case_patients,
        "document_records": total_document_records,
        "lit_document_records": lit_document_records,
        "emr_document_records": emr_document_records,
        "lit_document_id_fingerprint": lit_document_fingerprint,
        "emr_document_id_fingerprint": emr_document_fingerprint,
        "total_chunks": total_chunks,
        "lit_chunks": lit_chunks,
        "emr_chunks": emr_chunks,
        "active_embedding_chunks": active_chunks,
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_dimension": settings.active_embedding_dimension(),
        "corpus_embedding_input_type": "document",
        "chunk_id_fingerprint": mongo_fingerprint,
        "bm25_rows": fts_rows,
        "bm25_chunk_id_fingerprint": fts_fingerprint,
        "hnsw_vectors": hnsw_vectors,
        "hnsw_mapping_rows": mapping_rows,
        "hnsw_mapping_label_bounds": label_bounds,
        "hnsw_mapping_chunk_id_fingerprint": mapping_fingerprint,
        "hnsw_vector_alignment": hnsw_alignment,
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
    cases_value = args.cases_path or runtime.get("cases_path") or ""
    if not str(cases_value).strip():
        raise RuntimeError("No full TREC cases path is configured")
    result = verify(args.org_id, Path(str(cases_value)))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
