"""Shared identity constants for the official TREC CDS 2016 runtime."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCUMENTS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_QRELS = 37_707
EXPECTED_SAMPLEVAL_LINES = 108_012
EXPECTED_EMBEDDING = "embed-v-4-0"
EXPECTED_RERANKER = "Cohere-rerank-v4.0-fast"
EXPECTED_CHUNKING_CONTRACT = "app.medswin.chunking.section_chunks/full-body"
PREPARATION_CONTRACT_VERSION = "full-trec-full-body-doc-embed-fts5-hnsw-v2"
CHUNKER_SOURCE = REPO_ROOT / "app" / "medswin" / "chunking.py"

RRF_K = 60
RRF_K_BM25 = 4000
RRF_K_DENSE = 4000
CASCADE_RERANK = 300
RUN_DEPTH = 1000
TOPIC_IDS = tuple(range(1, 31))
DIAGNOSIS_TOPICS = tuple(range(1, 11))
TEST_TOPICS = tuple(range(11, 21))
TREATMENT_TOPICS = tuple(range(21, 31))

TABLE8_NOTE_MEDIAN_INFNDCG = 0.1228
TABLE8_NOTE_MEDIAN_P10 = 0.1833
TABLE10_SUMMARY_MEDIAN_INFNDCG = 0.1859
TABLE10_SUMMARY_MEDIAN_P10 = 0.2633

TYPE_QUESTIONS = {
    "diagnosis": "What is the patient's diagnosis?",
    "test": "What tests should the patient receive?",
    "treatment": "How should the patient be treated?",
}

SYSTEMS = ("bm25", "dense", "rrf", "cascade")
RUN_NAMES = {
    "bm25": "msbm25note",
    "dense": "msdensnote",
    "rrf": "msrrfnote",
    "cascade": "mscascnote",
}
SUMMARY_RUN_NAMES = {
    "bm25": "msbm25summ",
    "dense": "msdenssumm",
    "rrf": "msrrfsumm",
    "cascade": "mscascsumm",
}


@lru_cache(maxsize=1)
def chunker_sha256() -> str:
    if not CHUNKER_SOURCE.exists():
        raise RuntimeError(f"Chunker source is missing: {CHUNKER_SOURCE}")
    return hashlib.sha256(CHUNKER_SOURCE.read_bytes()).hexdigest()
