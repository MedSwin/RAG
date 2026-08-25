"""Shared identity constants for the strict complete-TREC evaluation runtime.

Keep this module intentionally lightweight. Verification, document
materialization, CI tests, and the expensive corpus builder all need to agree on
exactly what constitutes the same prepared corpus, but importing the builder
itself would also import ir_datasets and other heavy runtime dependencies.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DATASET = "pmc/v2/trec-cds-2016"
EXPECTED_DOCUMENTS = 1_255_260
EXPECTED_QUERIES = 30
EXPECTED_QRELS = 37_707
EXPECTED_EMBEDDING = "embed-v-4-0"
EXPECTED_RERANKER = "Cohere-rerank-v4.0-fast"
EXPECTED_CHUNKING_CONTRACT = "app.medswin.chunking.section_chunks/full-body"
PREPARATION_CONTRACT_VERSION = "full-trec-full-body-doc-embed-fts5-hnsw-v2"
CHUNKER_SOURCE = REPO_ROOT / "app" / "medswin" / "chunking.py"
ALL_PIPELINES: tuple[str, ...] = ("naive_rag", "medswin")
ALL_GENERATORS: tuple[str, ...] = ("medswin_local", "foundry")


def resolve_pipelines(pipeline: str) -> tuple[str, ...]:
    """Map the CLI pipeline selector onto the cells a matrix will execute."""
    if pipeline == "both":
        return ALL_PIPELINES
    if pipeline in ALL_PIPELINES:
        return (pipeline,)
    raise ValueError(f"Unsupported full-eval pipeline: {pipeline}")


def expected_matrix_keys(
    pipelines: tuple[str, ...] | list[str],
    generators: tuple[str, ...] | list[str] = ALL_GENERATORS,
) -> set[str]:
    return {f"{pipeline}:{generator}" for generator in generators for pipeline in pipelines}


def is_publication_matrix(
    pipelines: tuple[str, ...] | list[str],
    generators: tuple[str, ...] | list[str] = ALL_GENERATORS,
) -> bool:
    return tuple(pipelines) == ALL_PIPELINES and tuple(generators) == ALL_GENERATORS


@lru_cache(maxsize=1)
def chunker_sha256() -> str:
    """Return the exact source hash of the chunker used by publication prep."""
    if not CHUNKER_SOURCE.exists():
        raise RuntimeError(f"Chunker source is missing: {CHUNKER_SOURCE}")
    return hashlib.sha256(CHUNKER_SOURCE.read_bytes()).hexdigest()
