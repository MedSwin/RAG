#!/usr/bin/env python3
"""Idempotent MedSwin evaluation warmup.

Warmup invariants:
1. The exact MedSwin generator snapshot is present locally.
2. The complete ir_datasets TREC-CDS 2016 PMC corpus has been materialized and
   counted (not a judged pool or reservoir sample).
3. Azure Foundry chat, Cohere Embed v4, and Cohere Rerank v4 deployments are
   live through the same adapters used by the runtime.

The script writes machine-readable manifests under data/eval-warmup so later
full-corpus preparation can fail fast instead of silently running a smoke setup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_TREC_DOCS = 1_255_260
EXPECTED_TREC_QUERIES = 30
EXPECTED_TREC_QRELS = 37_707
DEFAULT_DATASET = "pmc/v2/trec-cds-2016"
DEFAULT_MODEL_ID = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "eval-warmup"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def warm_model(model_id: str, target: Path, force: bool) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".medswin_snapshot.json"
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if cached.get("model_id") == model_id and (target / "config.json").exists():
            print(f"[warmup] MedSwin model ready: {target}")
            return cached

    resolved = snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        token=os.getenv("HF_TOKEN") or None,
    )
    weight_files = sorted(
        str(path.relative_to(target))
        for pattern in ("*.safetensors", "*.bin")
        for path in target.rglob(pattern)
    )
    if not (target / "config.json").exists() or not weight_files:
        raise RuntimeError(f"Hugging Face snapshot at {target} is incomplete")
    payload = {
        "model_id": model_id,
        "target": str(target),
        "resolved_path": str(resolved),
        "weight_file_count": len(weight_files),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(marker, payload)
    print(f"[warmup] MedSwin model materialized: {target} ({len(weight_files)} weight files)")
    return payload


def warm_trec(dataset_name: str, force: bool) -> dict[str, Any]:
    import ir_datasets

    marker = MANIFEST_DIR / "trec-cds-2016-materialized.json"
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if (
            cached.get("dataset") == dataset_name
            and cached.get("docs") == EXPECTED_TREC_DOCS
            and cached.get("queries") == EXPECTED_TREC_QUERIES
            and cached.get("qrels") == EXPECTED_TREC_QRELS
        ):
            print(f"[warmup] Full TREC-CDS 2016 cache already verified: {EXPECTED_TREC_DOCS:,} docs")
            return cached

    dataset = ir_datasets.load(dataset_name)
    query_count = sum(1 for _ in dataset.queries_iter())
    qrel_count = sum(1 for _ in dataset.qrels_iter())
    if query_count != EXPECTED_TREC_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_TREC_QUERIES} TREC queries, found {query_count}")
    if qrel_count != EXPECTED_TREC_QRELS:
        raise RuntimeError(f"Expected {EXPECTED_TREC_QRELS} TREC qrels, found {qrel_count}")

    # Iterating the complete collection forces ir_datasets to download/cache the
    # PMC v2 collection and proves that every advertised document is readable.
    doc_count = 0
    for doc_count, _doc in enumerate(dataset.docs_iter(), start=1):
        if doc_count % 25_000 == 0:
            print(f"[warmup] TREC-CDS materialization verified {doc_count:,}/{EXPECTED_TREC_DOCS:,} docs")
    if doc_count != EXPECTED_TREC_DOCS:
        raise RuntimeError(f"Expected {EXPECTED_TREC_DOCS} PMC documents, found {doc_count}")

    payload = {
        "dataset": dataset_name,
        "docs": doc_count,
        "queries": query_count,
        "qrels": qrel_count,
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(marker, payload)
    print(f"[warmup] Full TREC-CDS 2016 materialized: {doc_count:,} docs")
    return payload


async def warm_foundry() -> dict[str, Any]:
    from app.core.config import settings
    from app.services.adapters.embedding import EmbeddingClient
    from app.services.adapters.llm import LLMClient
    from app.services.adapters.reranker import RerankerClient

    endpoint = (settings.AZURE_AI_FOUNDRY_ENDPOINT or "").strip()
    api_key = (settings.AZURE_AI_FOUNDRY_API_KEY or "").strip()
    if not endpoint or not api_key:
        raise RuntimeError(
            "AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY are required for evaluation warmup"
        )

    foundry_model = os.getenv("FOUNDRY_MODEL") or settings.CLOUD_MODEL or "gpt-5.4"
    expected_embedding = os.getenv("CLOUD_EMBEDDING") or settings.CLOUD_EMBEDDING
    expected_reranker = os.getenv("CLOUD_RERANKER") or settings.CLOUD_RERANKER

    embedding = EmbeddingClient(settings.active_embedding_url(), model=expected_embedding, api_key=api_key)
    reranker = RerankerClient(settings.active_reranker_url(), model=expected_reranker, api_key=api_key)

    previous_backend = os.environ.get("GENERATION_BACKEND")
    os.environ["GENERATION_BACKEND"] = "foundry"
    llm = LLMClient(settings.cloud_chat_url(), model=foundry_model, api_key=api_key)
    try:
        query_vec = (await embedding.embed(["aspirin myocardial infarction"], input_type="query"))[0]
        doc_vec = (await embedding.embed(["Aspirin reduces platelet aggregation."], input_type="document"))[0]
        expected_dim = settings.active_embedding_dimension()
        if len(query_vec) != expected_dim or len(doc_vec) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: query={len(query_vec)} document={len(doc_vec)} expected={expected_dim}"
            )
        ranks = await reranker.rerank(
            "aspirin antiplatelet therapy",
            ["Aspirin inhibits platelet aggregation.", "The moon orbits Earth."],
        )
        if len(ranks) != 2 or {item.get("index") for item in ranks} != {0, 1}:
            raise RuntimeError("Cohere reranker warmup returned an invalid result set")
        completion = await llm.call_llm(
            [{"role": "user", "content": "Reply with exactly: MEDSWIN_FOUNDRY_OK"}],
            max_tokens=32,
            temperature=0.0,
        )
        if not completion.get("content"):
            raise RuntimeError("Foundry GPT warmup returned empty content")
    finally:
        await embedding.close()
        await reranker.close()
        await llm.close()
        if previous_backend is None:
            os.environ.pop("GENERATION_BACKEND", None)
        else:
            os.environ["GENERATION_BACKEND"] = previous_backend

    payload = {
        "endpoint_host": endpoint.split("//", 1)[-1].split("/", 1)[0],
        "foundry_model": foundry_model,
        "embedding_model": expected_embedding,
        "embedding_dimension": expected_dim,
        "reranker_model": expected_reranker,
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(MANIFEST_DIR / "foundry-services.json", payload)
    print(
        "[warmup] Azure Foundry verified: "
        f"LLM={foundry_model}, embed={expected_embedding}/{expected_dim}, rerank={expected_reranker}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-trec", action="store_true")
    parser.add_argument("--skip-foundry", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-id", default=os.getenv("MEDSWIN_MODEL_REPO", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--model-path",
        default=os.getenv("MEDSWIN_MODEL_PATH", str(ROOT / "models" / "MedSwin-DaRE-TIES-KD-0.7")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if not args.skip_model:
        results["model"] = warm_model(args.model_id, Path(args.model_path), args.force)
    if not args.skip_trec:
        results["trec"] = warm_trec(args.dataset, args.force)
    if not args.skip_foundry:
        results["foundry"] = asyncio.run(warm_foundry())
    results["complete"] = all(
        bool(value.get("complete", True)) for key, value in results.items() if key != "complete"
    )
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(MANIFEST_DIR / "warmup.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
