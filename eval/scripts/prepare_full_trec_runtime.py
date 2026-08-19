#!/usr/bin/env python3
"""Prepare the complete TREC-CDS 2016 runtime corpus for publication evaluation.

Unlike ``ingest_trec_pmc.py`` this command has no sample mode and never truncates
PMC article bodies before MedSwin's section-aware chunker. It performs the exact
corpus preparation needed by the full evaluation matrix:

* all 1,255,260 ``pmc/v2/trec-cds-2016`` documents are read;
* title + abstract + complete body are chunked by ``app.medswin.chunking``;
* every materialized chunk is embedded with the active Foundry Cohere Embed v4
  deployment using ``input_type=document``;
* all 30 TREC case notes are materialized as patient-scoped EMR chunks in the
  same vector space;
* a disk-backed SQLite FTS5 BM25 corpus is built for the hybrid lexical stage;
* HNSW is built incrementally rather than first collecting every vector in a
  Python list; its label mapping is a lazy SQLite database;
* exact-count and embedding-space invariants are written to a strict manifest.

This is intentionally expensive. A valid publication run should prefer a
restartable, exact corpus build over a fast but ambiguous sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import hnswlib
import ir_datasets
import numpy as np
from pymongo import ReplaceOne

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
SCRIPTS_ROOT = EVAL_ROOT / "scripts"
for root in (REPO_ROOT, EVAL_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.config import settings
from app.core.database import get_sync_database
from app.medswin.chunking import section_chunks
from app.services.adapters.embedding import EmbeddingClient
from facets import benchmark_facet_templates

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_QRELS = 37_707
DEFAULT_ORG = "bench-org"
MANIFEST_DIR = REPO_ROOT / "data" / "eval-warmup"
DEFAULT_CASES = EVAL_ROOT / "data" / "trec-cds-2016" / "full" / "cases.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _warmup_manifest() -> dict[str, Any]:
    path = MANIFEST_DIR / "warmup.json"
    if not path.exists():
        raise RuntimeError("Missing data/eval-warmup/warmup.json; run scripts/warmup-eval.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trec = payload.get("trec") or {}
    foundry = payload.get("foundry") or {}
    if trec.get("docs") != EXPECTED_DOCS or not trec.get("complete"):
        raise RuntimeError("Warmup did not verify the complete TREC-CDS corpus")
    if foundry.get("embedding_model") != settings.CLOUD_EMBEDDING or not foundry.get("complete"):
        raise RuntimeError("Warmup did not verify the active Foundry embedding/reranker services")
    return payload


def _full_text(doc: Any) -> str:
    title = str(getattr(doc, "title", "") or "").strip()
    abstract = str(getattr(doc, "abstract", "") or "").strip()
    body = str(getattr(doc, "body", "") or "").strip()
    doc_id = str(getattr(doc, "doc_id"))
    sections: list[str] = []
    if title:
        sections.extend(["Title", title])
    if abstract:
        sections.extend(["Abstract", abstract])
    if body:
        sections.extend(["Body", body])
    # Never drop a collection member because its snapshot is metadata-only.
    if not sections:
        sections.extend(["Document", f"PMC document {doc_id}"])
    return "\n\n".join(sections)


def _topic_text(query: Any) -> str:
    for name in ("description", "summary", "note"):
        value = getattr(query, name, None)
        if value and str(value).strip():
            return str(value).strip()
    return f"Clinical question for topic {getattr(query, 'query_id', '')}"


def _patient_context(query: Any) -> str:
    for name in ("note", "description", "summary"):
        value = getattr(query, name, None)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _build_cases(dataset: Any, out_path: Path) -> tuple[list[dict[str, Any]], set[str], int]:
    qrels_by_qid: dict[str, list[str]] = defaultdict(list)
    relevance_by_qid: dict[str, dict[str, int]] = defaultdict(dict)
    total_qrels = 0
    for qrel in dataset.qrels_iter():
        total_qrels += 1
        if int(qrel.relevance) > 0:
            qid = str(qrel.query_id)
            doc_id = str(qrel.doc_id)
            qrels_by_qid[qid].append(doc_id)
            relevance_by_qid[qid][doc_id] = int(qrel.relevance)
    if total_qrels != EXPECTED_QRELS:
        raise RuntimeError(f"Expected {EXPECTED_QRELS} qrels, found {total_qrels}")

    rows: list[dict[str, Any]] = []
    gold_doc_ids: set[str] = set()
    for query in dataset.queries_iter():
        qid = str(getattr(query, "query_id"))
        qtype = str(getattr(query, "type", "clinical") or "clinical").lower()
        gold_docs = sorted(set(qrels_by_qid.get(qid, [])))
        gold_doc_ids.update(gold_docs)
        facets = benchmark_facet_templates(qtype)
        for facet in facets:
            facet["gold_doc_ids"] = gold_docs
            facet["notes"] = "Auto-seeded from the complete positive TREC CDS qrels; no per-topic cap."
        rows.append(
            {
                "case_id": qid,
                "dataset": EXPECTED_DATASET,
                "query": _topic_text(query),
                "query_type": qtype,
                "patient_id": f"trec-cds-{qid}",
                "patient_context": _patient_context(query),
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
                    "source_dataset": EXPECTED_DATASET,
                    "qrels_complete": True,
                },
            }
        )
    if len(rows) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} queries, found {len(rows)}")
    _write_jsonl(out_path, rows)
    return rows, gold_doc_ids, total_qrels


def _checkpoint_path(org_id: str) -> Path:
    return MANIFEST_DIR / f"full-trec-{org_id}-checkpoint.json"


def _load_checkpoint(org_id: str) -> dict[str, Any]:
    path = _checkpoint_path(org_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_checkpoint(checkpoint: dict[str, Any], org_id: str) -> int:
    if not checkpoint:
        return 0
    expected = {
        "dataset": EXPECTED_DATASET,
        "org_id": org_id,
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_dim": settings.active_embedding_dimension(),
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
    }
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "Full-corpus checkpoint does not match the active evaluation configuration: "
            + ", ".join(mismatches)
            + ". Reset the benchmark org before rebuilding."
        )
    return int(checkpoint.get("next_doc_ordinal") or 0)


def _reset(org_id: str) -> None:
    db = get_sync_database()
    for name in ("chunks", "documents", "traces", "sessions"):
        deleted = db[name].delete_many({"org_id": org_id}).deleted_count
        print(f"[full-prep] reset {name}: {deleted:,}")
    for raw in (
        settings.HNSW_INDEX_PATH,
        settings.HNSW_MAPPING_PATH,
        settings.FAISS_INDEX_PATH,
        settings.FAISS_MAPPING_PATH,
        settings.TREE_INDEX_PATH,
        settings.TREE_MAPPING_PATH,
    ):
        path = Path(raw)
        if path.exists():
            path.unlink()
        sidecar = path.with_name(f"{path.name}.manifest.json")
        if sidecar.exists():
            sidecar.unlink()
    fts = Path(os.getenv("LEXICAL_FTS_PATH") or (Path(settings.DATA_DIR) / "bm25" / f"{org_id}_fts.sqlite"))
    if fts.exists():
        fts.unlink()
    checkpoint = _checkpoint_path(org_id)
    if checkpoint.exists():
        checkpoint.unlink()


def _chunk_record(org_id: str, doc_id: str, title: str, chunk: dict[str, Any], ordinal: int) -> dict[str, Any]:
    text = str(chunk.get("text") or chunk.get("content") or "").strip()
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "doc_id": doc_id,
        "source_type": "LIT",
        "text": text,
        "section": chunk.get("section"),
        "offset_start": chunk.get("offset_start"),
        "offset_end": chunk.get("offset_end"),
        "patient_id": None,
        "org_id": org_id,
        "evidence_grade": {
            "label": "biomedical_literature",
            "score": 0.65,
            "source_reliability": 0.75,
        },
        "source_reliability": 0.75,
        "metadata": {
            "title": title,
            "dataset": EXPECTED_DATASET,
            "dataset_doc_ordinal": ordinal,
            **(chunk.get("metadata") or {}),
        },
        "tokenized_text": text.lower().split(),
        "created_at": datetime.now(timezone.utc),
    }


async def _embed_records(
    client: EmbeddingClient,
    records: list[dict[str, Any]],
    batch_size: int,
    delay_s: float,
) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = await client.embed([record["text"] for record in batch], input_type="document")
        if len(vectors) != len(batch):
            raise RuntimeError(f"Embedding response length mismatch at chunk batch {start}")
        for record, vector in zip(batch, vectors):
            if len(vector) != settings.active_embedding_dimension():
                raise RuntimeError(
                    f"Embedding dimension {len(vector)} != active {settings.active_embedding_dimension()}"
                )
            record["embedding"] = vector.tolist()
            record["embedding_model"] = settings.active_embedding_model()
            record["embedding_dim"] = int(len(vector))
            record["embedding_space"] = settings.active_embedding_space()
            record["embedding_updated_at"] = datetime.now(timezone.utc)
        if delay_s > 0 and start + batch_size < len(records):
            await asyncio.sleep(delay_s)


def _persist_records(org_id: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    coll = get_sync_database()["chunks"]
    operations = [
        ReplaceOne({"org_id": org_id, "chunk_id": record["chunk_id"]}, record, upsert=True)
        for record in records
    ]
    coll.bulk_write(operations, ordered=False)


async def _ingest_literature(
    dataset: Any,
    org_id: str,
    start_ordinal: int,
    embed_batch_size: int,
    flush_chunk_count: int,
    delay_s: float,
) -> dict[str, int]:
    checkpoint_path = _checkpoint_path(org_id)
    client = EmbeddingClient(settings.active_embedding_url())
    pending: list[dict[str, Any]] = []
    last_ordinal = start_ordinal - 1
    docs_seen = start_ordinal
    chunks_written = 0
    flush_last_ordinal = start_ordinal - 1

    async def flush() -> None:
        nonlocal pending, chunks_written, flush_last_ordinal
        if not pending:
            return
        await _embed_records(client, pending, embed_batch_size, delay_s)
        _persist_records(org_id, pending)
        chunks_written += len(pending)
        pending = []
        checkpoint = {
            "dataset": EXPECTED_DATASET,
            "org_id": org_id,
            "embedding_model": settings.active_embedding_model(),
            "embedding_space": settings.active_embedding_space(),
            "embedding_dim": settings.active_embedding_dimension(),
            "target_chunk_size": settings.TARGET_CHUNK_SIZE,
            "next_doc_ordinal": flush_last_ordinal + 1,
            "completed": False,
            "updated_at": _now(),
        }
        _atomic_json(checkpoint_path, checkpoint)

    try:
        for ordinal, doc in enumerate(dataset.docs_iter()):
            if ordinal < start_ordinal:
                continue
            doc_id = str(getattr(doc, "doc_id"))
            title = str(getattr(doc, "title", "") or "Untitled PMC article")
            text = _full_text(doc)
            raw_chunks = section_chunks(f"{org_id}:{doc_id}", text, default_section="PMC article")
            if not raw_chunks:
                raise RuntimeError(f"PMC document {doc_id} produced no chunks")
            for raw in raw_chunks:
                pending.append(_chunk_record(org_id, doc_id, title, raw, ordinal))
            last_ordinal = ordinal
            flush_last_ordinal = ordinal
            docs_seen = ordinal + 1
            if len(pending) >= flush_chunk_count:
                await flush()
                if docs_seen % 1000 == 0 or docs_seen == EXPECTED_DOCS:
                    print(
                        f"[full-prep] literature {docs_seen:,}/{EXPECTED_DOCS:,} docs; "
                        f"new chunks={chunks_written:,}",
                        flush=True,
                    )
        await flush()
    finally:
        await client.close()

    if docs_seen != EXPECTED_DOCS or last_ordinal != EXPECTED_DOCS - 1:
        raise RuntimeError(f"Full corpus ended at {docs_seen:,} documents; expected {EXPECTED_DOCS:,}")
    checkpoint = {
        "dataset": EXPECTED_DATASET,
        "org_id": org_id,
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_dim": settings.active_embedding_dimension(),
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
        "next_doc_ordinal": EXPECTED_DOCS,
        "completed": True,
        "updated_at": _now(),
    }
    _atomic_json(checkpoint_path, checkpoint)
    return {"documents": docs_seen, "new_chunks": chunks_written}


async def _ingest_case_notes(dataset: Any, org_id: str, embed_batch_size: int, delay_s: float) -> int:
    records: list[dict[str, Any]] = []
    for query in dataset.queries_iter():
        qid = str(getattr(query, "query_id"))
        patient_id = f"trec-cds-{qid}"
        text = _patient_context(query) or _topic_text(query)
        raw_chunks = section_chunks(f"{org_id}:trec-cds-2016:{qid}:note", text, default_section="patient_context")
        for chunk in raw_chunks:
            body = str(chunk.get("text") or "").strip()
            records.append(
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "doc_id": f"trec-cds-2016:{qid}:note",
                    "source_type": "EMR",
                    "text": body,
                    "section": chunk.get("section"),
                    "offset_start": chunk.get("offset_start"),
                    "offset_end": chunk.get("offset_end"),
                    "patient_id": patient_id,
                    "org_id": org_id,
                    "evidence_grade": {
                        "label": "emr_note",
                        "score": 0.70,
                        "source_reliability": 0.80,
                    },
                    "source_reliability": 0.80,
                    "metadata": {
                        "benchmark_case_id": qid,
                        "dataset": EXPECTED_DATASET,
                        "benchmark_patient_context": True,
                    },
                    "tokenized_text": body.lower().split(),
                    "created_at": datetime.now(timezone.utc),
                }
            )
    client = EmbeddingClient(settings.active_embedding_url())
    try:
        await _embed_records(client, records, embed_batch_size, delay_s)
    finally:
        await client.close()
    _persist_records(org_id, records)
    print(f"[full-prep] case-note EMR chunks={len(records):,}")
    return len(records)


def _fts_path(org_id: str) -> Path:
    explicit = (os.getenv("LEXICAL_FTS_PATH") or "").strip()
    if explicit:
        return Path(explicit.format(org_id=org_id))
    return Path(settings.DATA_DIR) / "bm25" / f"{org_id}_fts.sqlite"


def _build_fts(org_id: str, batch_size: int) -> dict[str, Any]:
    path = _fts_path(org_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_id UNINDEXED, org_id UNINDEXED, source_type UNINDEXED, "
                "patient_id UNINDEXED, text, tokenize='porter')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Python SQLite must include FTS5 support for full-corpus BM25") from exc

        coll = get_sync_database()["chunks"]
        cursor = coll.find(
            {"org_id": org_id},
            {"_id": 0, "chunk_id": 1, "org_id": 1, "source_type": 1, "patient_id": 1, "text": 1},
            batch_size=batch_size,
        )
        rows: list[tuple[Any, ...]] = []
        inserted = 0
        for chunk in cursor:
            rows.append(
                (
                    str(chunk.get("chunk_id")),
                    org_id,
                    str(chunk.get("source_type") or ""),
                    str(chunk.get("patient_id") or ""),
                    str(chunk.get("text") or ""),
                )
            )
            if len(rows) >= batch_size:
                conn.executemany(
                    "INSERT INTO chunks_fts(chunk_id, org_id, source_type, patient_id, text) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                inserted += len(rows)
                rows = []
                if inserted % 100_000 == 0:
                    print(f"[full-prep] BM25 FTS rows={inserted:,}", flush=True)
        if rows:
            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, org_id, source_type, patient_id, text) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            inserted += len(rows)
        conn.commit()
        row_count = int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        if row_count != inserted:
            raise RuntimeError(f"FTS row count {row_count} != inserted {inserted}")
        # Smoke the actual bm25() function, not only table creation.
        conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
            ('"clinical"',),
        ).fetchall()
    finally:
        conn.close()
    return {"path": str(path), "rows": inserted, "backend": "sqlite-fts5-bm25"}


def _mapping_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE mapping(label INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE)")
    return conn


def _build_hnsw(org_id: str, mongo_batch_size: int) -> dict[str, Any]:
    coll = get_sync_database()["chunks"]
    active_filter = {
        "org_id": org_id,
        "embedding_space": settings.active_embedding_space(),
        "embedding_model": settings.active_embedding_model(),
        "embedding_dim": settings.active_embedding_dimension(),
        "embedding": {"$exists": True, "$ne": []},
    }
    total_vectors = coll.count_documents(active_filter)
    total_chunks = coll.count_documents({"org_id": org_id})
    if total_vectors != total_chunks or total_vectors <= 0:
        raise RuntimeError(
            f"Cannot build full index: active vectors={total_vectors:,}, total chunks={total_chunks:,}"
        )

    index_path = Path(settings.HNSW_INDEX_PATH)
    mapping_path = Path(settings.HNSW_MAPPING_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()

    dim = settings.active_embedding_dimension()
    index = hnswlib.Index(space="cosine", dim=dim)
    m = int(os.getenv("FULL_HNSW_M", "32"))
    ef_construction = int(os.getenv("FULL_HNSW_EF_CONSTRUCTION", str(max(384, m * 12))))
    index.init_index(max_elements=total_vectors, ef_construction=ef_construction, M=m)

    mapping = _mapping_db(mapping_path)
    cursor = coll.find(active_filter, {"_id": 0, "chunk_id": 1, "embedding": 1}, batch_size=mongo_batch_size)
    labels_written = 0
    batch_vectors: list[list[float]] = []
    batch_ids: list[str] = []

    def flush() -> None:
        nonlocal labels_written, batch_vectors, batch_ids
        if not batch_vectors:
            return
        matrix = np.asarray(batch_vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != dim:
            raise RuntimeError(f"Invalid HNSW batch shape {matrix.shape}; expected (*, {dim})")
        labels = np.arange(labels_written, labels_written + len(batch_ids), dtype=np.int64)
        index.add_items(matrix, labels)
        mapping.executemany(
            "INSERT INTO mapping(label, chunk_id) VALUES (?, ?)",
            [(int(label), chunk_id) for label, chunk_id in zip(labels, batch_ids)],
        )
        labels_written += len(batch_ids)
        batch_vectors = []
        batch_ids = []
        if labels_written % 100_000 == 0:
            mapping.commit()
            print(f"[full-prep] HNSW vectors={labels_written:,}/{total_vectors:,}", flush=True)

    try:
        for chunk in cursor:
            vector = chunk.get("embedding") or []
            if len(vector) != dim:
                raise RuntimeError(f"Chunk {chunk.get('chunk_id')} has dimension {len(vector)}; expected {dim}")
            batch_vectors.append(vector)
            batch_ids.append(str(chunk["chunk_id"]))
            if len(batch_vectors) >= mongo_batch_size:
                flush()
        flush()
        mapping.commit()
        mapping_rows = int(mapping.execute("SELECT count(*) FROM mapping").fetchone()[0])
    finally:
        mapping.close()

    if labels_written != total_vectors or mapping_rows != total_vectors:
        raise RuntimeError(
            f"HNSW completeness mismatch vectors={labels_written}, mapping={mapping_rows}, expected={total_vectors}"
        )
    index.set_ef(min(max(200, settings.HNSW_MAX_EF_SEARCH), total_vectors))
    index.save_index(str(index_path))

    source_counts = {
        source: coll.count_documents({**active_filter, "source_type": source})
        for source in ("LIT", "EMR", "CPG", "SAFETY")
    }
    manifest = {
        "org_id": org_id,
        "dataset": EXPECTED_DATASET,
        "index_type": "hnsw",
        "index_path": str(index_path),
        "mapping_path": str(mapping_path),
        "mapping_backend": "sqlite",
        "embedding_space": settings.active_embedding_space(),
        "embedding_model": settings.active_embedding_model(),
        "embedding_dim": dim,
        "total_vectors": total_vectors,
        "chunk_count": total_chunks,
        "source_counts": source_counts,
        "expected_trec_documents": EXPECTED_DOCS,
        "M": m,
        "ef_construction": ef_construction,
        "built_at": _now(),
        "full_corpus": True,
    }
    manifest_path = index_path.with_name(f"{index_path.name}.manifest.json")
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _verify_mongo(org_id: str, gold_doc_ids: set[str]) -> dict[str, Any]:
    coll = get_sync_database()["chunks"]
    base = {"org_id": org_id}
    lit_chunks = coll.count_documents({**base, "source_type": "LIT"})
    emr_chunks = coll.count_documents({**base, "source_type": "EMR"})
    active = {
        **base,
        "embedding_space": settings.active_embedding_space(),
        "embedding_model": settings.active_embedding_model(),
        "embedding_dim": settings.active_embedding_dimension(),
        "embedding": {"$exists": True, "$ne": []},
    }
    total_chunks = coll.count_documents(base)
    active_chunks = coll.count_documents(active)
    stale = total_chunks - active_chunks
    gold_present = set(
        str(value)
        for value in coll.distinct("doc_id", {**base, "source_type": "LIT", "doc_id": {"$in": list(gold_doc_ids)}})
    )
    if stale:
        raise RuntimeError(f"Full corpus contains {stale:,} chunks outside the active embedding space")
    if gold_present != gold_doc_ids:
        missing = sorted(gold_doc_ids - gold_present)
        raise RuntimeError(f"Missing {len(missing)} positive qrel documents from corpus; sample={missing[:20]}")
    return {
        "total_chunks": total_chunks,
        "lit_chunks": lit_chunks,
        "emr_chunks": emr_chunks,
        "active_chunks": active_chunks,
        "stale_chunks": stale,
        "positive_qrel_doc_count": len(gold_doc_ids),
        "positive_qrel_docs_present": len(gold_present),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset != EXPECTED_DATASET:
        raise RuntimeError(f"Full evaluation requires {EXPECTED_DATASET}; got {args.dataset}")
    if not settings.CLOUD_MODE:
        raise RuntimeError("Full TREC runtime preparation requires CLOUD_MODE=true")
    if settings.CLOUD_EMBEDDING != "embed-v-4-0":
        raise RuntimeError(f"Expected CLOUD_EMBEDDING=embed-v-4-0; got {settings.CLOUD_EMBEDDING}")
    if settings.active_embedding_dimension() not in {256, 512, 1024, 1536}:
        raise RuntimeError("embed-v-4-0 dimension must be one of 256, 512, 1024, 1536")
    _warmup_manifest()

    if args.reset:
        _reset(args.org_id)
    checkpoint = _load_checkpoint(args.org_id)
    start_ordinal = _validate_checkpoint(checkpoint, args.org_id)
    db = get_sync_database()
    existing_chunks = db["chunks"].count_documents({"org_id": args.org_id})
    if start_ordinal == 0 and existing_chunks and not args.reset:
        raise RuntimeError(
            f"Benchmark org {args.org_id} already has {existing_chunks:,} chunks but no compatible checkpoint. "
            "Use --reset to avoid mixing corpora."
        )

    dataset = ir_datasets.load(args.dataset)
    cases, gold_doc_ids, total_qrels = _build_cases(dataset, Path(args.cases_out))
    literature = {"documents": EXPECTED_DOCS, "new_chunks": 0}
    if start_ordinal < EXPECTED_DOCS:
        literature = await _ingest_literature(
            dataset,
            args.org_id,
            start_ordinal,
            args.embed_batch_size,
            args.flush_chunks,
            args.inter_batch_delay,
        )
    else:
        print("[full-prep] literature checkpoint already complete; reusing corpus")

    emr_chunks = await _ingest_case_notes(
        dataset,
        args.org_id,
        args.embed_batch_size,
        args.inter_batch_delay,
    )
    mongo = _verify_mongo(args.org_id, gold_doc_ids)
    fts = _build_fts(args.org_id, args.disk_batch_size)
    if fts["rows"] != mongo["total_chunks"]:
        raise RuntimeError(f"BM25 FTS rows={fts['rows']} != corpus chunks={mongo['total_chunks']}")
    hnsw = _build_hnsw(args.org_id, args.disk_batch_size)
    if hnsw["total_vectors"] != mongo["total_chunks"]:
        raise RuntimeError("HNSW did not index every active corpus chunk")

    manifest = {
        "complete": True,
        "dataset": EXPECTED_DATASET,
        "expected_documents": EXPECTED_DOCS,
        "queries": len(cases),
        "qrels": total_qrels,
        "org_id": args.org_id,
        "cases_path": str(Path(args.cases_out).resolve()),
        "embedding_model": settings.active_embedding_model(),
        "embedding_space": settings.active_embedding_space(),
        "embedding_dim": settings.active_embedding_dimension(),
        "embedding_input_type": "document",
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
        "chunking": "app.medswin.chunking.section_chunks/full-body",
        "literature": literature,
        "emr_chunks": emr_chunks,
        "mongo": mongo,
        "bm25": fts,
        "hnsw": hnsw,
        "completed_at": _now(),
    }
    output = MANIFEST_DIR / f"full-trec-runtime-{args.org_id}.json"
    _atomic_json(output, manifest)
    print(f"[full-prep] COMPLETE manifest={output}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=EXPECTED_DATASET)
    parser.add_argument("--org-id", default=os.getenv("BENCHMARK_ORG_ID", DEFAULT_ORG))
    parser.add_argument("--cases-out", default=str(DEFAULT_CASES))
    parser.add_argument("--reset", action="store_true", help="Delete the benchmark org and rebuild from zero.")
    parser.add_argument("--embed-batch-size", type=int, default=int(os.getenv("FULL_EMBED_BATCH_SIZE", "64")))
    parser.add_argument("--flush-chunks", type=int, default=int(os.getenv("FULL_INGEST_FLUSH_CHUNKS", "512")))
    parser.add_argument("--disk-batch-size", type=int, default=int(os.getenv("FULL_DISK_BATCH_SIZE", "2048")))
    parser.add_argument("--inter-batch-delay", type=float, default=float(os.getenv("FULL_EMBED_BATCH_DELAY_S", "0")))
    args = parser.parse_args()
    if args.embed_batch_size <= 0 or args.flush_chunks <= 0 or args.disk_batch_size <= 0:
        parser.error("batch sizes must be positive")
    return args


def main() -> int:
    started = time.time()
    manifest = asyncio.run(run(parse_args()))
    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "documents": manifest["expected_documents"],
                "chunks": manifest["mongo"]["total_chunks"],
                "vectors": manifest["hnsw"]["total_vectors"],
                "elapsed_seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
