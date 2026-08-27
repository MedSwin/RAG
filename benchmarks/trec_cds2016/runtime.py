"""Bind publication index artifacts and validate the local 7B server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPECTED_LOCAL_MODEL = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"


def runtime_artifact_env(manifest: dict[str, Any]) -> dict[str, str]:
    """Return child-process paths bound to the verified publication artifacts."""
    hnsw = manifest.get("hnsw") or {}
    bm25 = manifest.get("bm25") or {}
    index_path = Path(str(hnsw.get("index_path") or "")).resolve()
    mapping_path = Path(str(hnsw.get("mapping_path") or "")).resolve()
    fts_path = Path(str(bm25.get("path") or "")).resolve()
    if not index_path.exists() or not mapping_path.exists() or not fts_path.exists():
        raise RuntimeError("Verified publication retrieval artifacts disappeared before startup")
    artifact_dir = index_path.parent
    return {
        "HNSW_INDEX_PATH": str(index_path),
        "HNSW_MAPPING_PATH": str(mapping_path),
        "LEXICAL_FTS_PATH": str(fts_path),
        "FAISS_INDEX_PATH": str(artifact_dir / "faiss_unused.bin"),
        "FAISS_MAPPING_PATH": str(artifact_dir / "faiss_unused.json"),
        "TREE_INDEX_PATH": str(artifact_dir / "tree_unused.npy"),
        "TREE_MAPPING_PATH": str(artifact_dir / "tree_unused.json"),
    }


def validate_local_health(health: dict[str, Any], expected_revision: str) -> None:
    if health.get("model") != EXPECTED_LOCAL_MODEL:
        raise RuntimeError(f"Local generator health reported unexpected model {health.get('model')}")
    if str(health.get("model_revision") or "").strip() != expected_revision:
        raise RuntimeError(
            f"Local generator revision {health.get('model_revision')!r} does not match warmup revision {expected_revision}"
        )
    if int(health.get("context_window") or 0) <= 0:
        raise RuntimeError("Local MedSwin server did not report a context window")
    if health.get("prompt_policy") != "fail_on_overflow_no_truncation":
        raise RuntimeError("Local MedSwin server is not enforcing fail-on-overflow/no-truncation")
