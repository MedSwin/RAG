"""Official TREC CDS 2016 retrieval evaluation (NIST sample_eval / trec_eval)."""

from .contract import (
    EXPECTED_CHUNKING_CONTRACT,
    EXPECTED_DATASET,
    EXPECTED_DOCUMENTS,
    EXPECTED_QRELS,
    EXPECTED_QUERIES,
    PREPARATION_CONTRACT_VERSION,
    chunker_sha256,
)

__all__ = [
    "EXPECTED_CHUNKING_CONTRACT",
    "EXPECTED_DATASET",
    "EXPECTED_DOCUMENTS",
    "EXPECTED_QRELS",
    "EXPECTED_QUERIES",
    "PREPARATION_CONTRACT_VERSION",
    "chunker_sha256",
]
