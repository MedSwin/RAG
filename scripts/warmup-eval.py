#!/usr/bin/env python3
"""Idempotent MedSwin system-evaluation warmup.

Warmup invariants:
1. The exact MedSwin generator snapshot is present locally and pinned to a
   concrete Hugging Face revision.
2. The complete ir_datasets TREC-CDS 2016 PMC corpus has been materialized and
   exact-count verified (not a judged pool or reservoir sample).
3. The exact requested Azure Foundry GPT, Cohere Embed v4, and Cohere Rerank v4
   deployments are live through the same adapters used by the runtime.

The script writes machine-readable manifests under ``data/eval-warmup``. Every
probe is fail-closed: a non-empty-but-wrong GPT response, wrong model identity,
wrong embedding dimension, malformed rerank result, or incomplete corpus does
not count as a successful warmup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.trec_cds2016.contract import EXPECTED_DOCUMENTS, EXPECTED_QRELS, EXPECTED_QUERIES
from benchmarks.trec_cds2016.nist import download_nist, ensure_trec_eval, verify_nist

EXPECTED_TREC_DOCS = EXPECTED_DOCUMENTS
EXPECTED_TREC_QUERIES = EXPECTED_QUERIES
EXPECTED_TREC_QRELS = EXPECTED_QRELS
DEFAULT_DATASET = "pmc/v2/trec-cds-2016"
DEFAULT_MODEL_ID = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
EXPECTED_FOUNDRY_MODEL = "gpt-5.4"
EXPECTED_EMBEDDING_MODEL = "embed-v-4-0"
EXPECTED_RERANKER_MODEL = "Cohere-rerank-v4.0-fast"
FOUNDRY_PROBE_MARKER = "MEDSWIN_FOUNDRY_OK"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "eval-warmup"
NLI_MODEL = "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"
NLI_REVISION = "e9c646143e4b88035c7a5505dc4dfaba59656bfd"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _weight_files(target: Path) -> list[str]:
    return sorted(
        str(path.relative_to(target))
        for pattern in ("*.safetensors", "*.bin")
        for path in target.rglob(pattern)
        if path.is_file()
    )


def warm_model(model_id: str, target: Path, force: bool) -> dict[str, Any]:
    """Materialize and pin the exact local MedSwin generator snapshot."""
    from huggingface_hub import HfApi, snapshot_download

    if model_id != DEFAULT_MODEL_ID:
        raise RuntimeError(
            f"System-evaluation warmup requires {DEFAULT_MODEL_ID}; got {model_id}. "
            "Use a separate experiment profile for other generator models."
        )

    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".medswin_snapshot.json"
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        cached_weights = _weight_files(target)
        if (
            cached.get("complete") is True
            and cached.get("model_id") == model_id
            and str(cached.get("revision") or "").strip()
            and (target / "config.json").exists()
            and cached_weights
        ):
            cached["weight_file_count"] = len(cached_weights)
            print(
                f"[warmup] MedSwin model ready: {target} "
                f"revision={cached.get('revision')} weights={len(cached_weights)}"
            )
            return cached

    token = os.getenv("HF_TOKEN") or None
    model_info = HfApi(token=token).model_info(model_id)
    revision = str(model_info.sha or "").strip()
    if not revision:
        raise RuntimeError(f"Hugging Face did not return a concrete revision for {model_id}")
    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(target),
        token=token,
    )
    weight_files = _weight_files(target)
    if not (target / "config.json").exists() or not weight_files:
        raise RuntimeError(f"Hugging Face snapshot at {target} is incomplete")
    payload = {
        "model_id": model_id,
        "revision": revision,
        "target": str(target),
        "resolved_path": str(resolved),
        "weight_file_count": len(weight_files),
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(marker, payload)
    print(
        f"[warmup] MedSwin model materialized: {target} "
        f"revision={revision} ({len(weight_files)} weight files)"
    )
    return payload


def warm_nli(force: bool) -> dict[str, Any]:
    """Materialize the fixed local evaluator without using either generator."""
    from huggingface_hub import snapshot_download
    target = ROOT / "models" / "system-eval-nli"
    marker = target / ".nli_snapshot.json"
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if cached.get("model") == NLI_MODEL and cached.get("revision") == NLI_REVISION and (target / "config.json").exists():
            return cached
    snapshot_download(repo_id=NLI_MODEL, revision=NLI_REVISION, local_dir=str(target), token=os.getenv("HF_TOKEN") or None)
    if not (target / "config.json").exists():
        raise RuntimeError("Pinned BioMedNLI evaluator snapshot is incomplete")
    payload = {"model": NLI_MODEL, "revision": NLI_REVISION, "path": str(target), "complete": True}
    _write_json(marker, payload)
    return payload


def warm_trec(dataset_name: str, force: bool) -> dict[str, Any]:
    """Materialize and exact-count verify the complete TREC-CDS 2016 collection."""
    import ir_datasets

    if dataset_name != DEFAULT_DATASET:
        raise RuntimeError(f"System-evaluation warmup requires {DEFAULT_DATASET}; got {dataset_name}")

    marker = MANIFEST_DIR / "trec-cds-2016-materialized.json"
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if (
            cached.get("complete") is True
            and cached.get("dataset") == dataset_name
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
    # PMC v2 collection and proves every advertised collection member is readable.
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
    """Probe the active cloud plane through the production adapters."""
    from app.core.config import settings

    if settings.cloud_provider() == "bedrock":
        return await warm_bedrock()
    return await _warm_foundry_azure()


async def warm_bedrock() -> dict[str, Any]:
    """Probe GPT-6 Luna, Cohere Embed v4, and Cohere Rerank 3.5 on Bedrock."""
    from app.core.config import settings
    from app.services.adapters.embedding import EmbeddingClient
    from app.services.adapters.llm import LLMClient
    from app.services.adapters.reranker import RerankerClient

    chat_model = settings.BEDROCK_CHAT_MODEL
    embed_model = settings.BEDROCK_EMBED_MODEL
    rerank_model = settings.BEDROCK_RERANK_MODEL
    if chat_model != "global.openai.gpt-6-luna":
        raise RuntimeError(f"Expected BEDROCK_CHAT_MODEL=global.openai.gpt-6-luna; got {chat_model}")
    if embed_model != "global.cohere.embed-v4:0":
        raise RuntimeError(f"Expected BEDROCK_EMBED_MODEL=global.cohere.embed-v4:0; got {embed_model}")
    if rerank_model != "cohere.rerank-v3-5:0":
        raise RuntimeError(f"Expected BEDROCK_RERANK_MODEL=cohere.rerank-v3-5:0; got {rerank_model}")

    embedding = EmbeddingClient(settings.active_embedding_url())
    reranker = RerankerClient(settings.active_reranker_url())
    previous_backend = os.environ.get("GENERATION_BACKEND")
    os.environ["GENERATION_BACKEND"] = "bedrock"
    llm = LLMClient(settings.active_llm_url("http://127.0.0.1:8000/v1/chat/completions"))
    marker = "MEDSWIN_BEDROCK_OK"
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
        if int(ranks[0].get("index", -1)) != 0:
            raise RuntimeError("Bedrock reranker did not rank the relevant passage first")
        completion = await llm.call_llm(
            [{"role": "user", "content": f"Reply with exactly: {marker}"}],
            max_tokens=16,
            temperature=0.0,
        )
        completion_text = str(completion.get("content") or "").strip()
        if completion_text != marker:
            raise RuntimeError(f"Bedrock probe returned {completion_text!r}; expected {marker!r}")
    finally:
        await embedding.close()
        await reranker.close()
        await llm.close()
        if previous_backend is None:
            os.environ.pop("GENERATION_BACKEND", None)
        else:
            os.environ["GENERATION_BACKEND"] = previous_backend

    payload = {
        "provider": "bedrock",
        "region": settings.AWS_BEDROCK_REGION,
        "chat_model": chat_model,
        "embedding_model": embed_model,
        "embedding_dimension": expected_dim,
        "embedding_input_types_verified": ["query", "document"],
        "reranker_model": rerank_model,
        "probe_marker": marker,
        "cloud_probe_exact": True,
        "foundry_probe_exact": True,
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(MANIFEST_DIR / "bedrock-services.json", payload)
    print(f"[warmup] Bedrock verified: LLM={chat_model}, embed={embed_model}/{expected_dim}, rerank={rerank_model}")
    return payload


async def _warm_foundry_azure() -> dict[str, Any]:
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

    foundry_model = os.getenv("FOUNDRY_MODEL") or settings.FOUNDRY_MODEL or settings.CLOUD_MODEL
    expected_embedding = os.getenv("CLOUD_EMBEDDING") or settings.CLOUD_EMBEDDING
    expected_reranker = os.getenv("CLOUD_RERANKER") or settings.CLOUD_RERANKER
    if foundry_model != EXPECTED_FOUNDRY_MODEL:
        raise RuntimeError(f"Expected FOUNDRY_MODEL={EXPECTED_FOUNDRY_MODEL}; got {foundry_model}")
    if expected_embedding != EXPECTED_EMBEDDING_MODEL:
        raise RuntimeError(f"Expected CLOUD_EMBEDDING={EXPECTED_EMBEDDING_MODEL}; got {expected_embedding}")
    if expected_reranker != EXPECTED_RERANKER_MODEL:
        raise RuntimeError(f"Expected CLOUD_RERANKER={EXPECTED_RERANKER_MODEL}; got {expected_reranker}")

    embedding = EmbeddingClient(settings.active_embedding_url(), model=expected_embedding, api_key=api_key)
    reranker = RerankerClient(settings.active_reranker_url(), model=expected_reranker, api_key=api_key)

    previous_backend = os.environ.get("GENERATION_BACKEND")
    os.environ["GENERATION_BACKEND"] = "foundry"
    llm = LLMClient(settings.cloud_chat_url(), model=foundry_model, api_key=api_key)
    try:
        # Both semantic intents are mandatory: corpus chunks use document mode,
        # while online retrieval uses query mode. A route that only supports one
        # of them cannot be certified for the complete evaluation.
        query_vec = (await embedding.embed(["aspirin myocardial infarction"], input_type="query"))[0]
        doc_vec = (await embedding.embed(["Aspirin reduces platelet aggregation."], input_type="document"))[0]
        expected_dim = settings.active_embedding_dimension()
        if len(query_vec) != expected_dim or len(doc_vec) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: query={len(query_vec)} document={len(doc_vec)} expected={expected_dim}"
            )
        if not any(float(value) != 0.0 for value in query_vec) or not any(float(value) != 0.0 for value in doc_vec):
            raise RuntimeError("Embedding warmup returned an all-zero vector")

        ranks = await reranker.rerank(
            "aspirin antiplatelet therapy",
            ["Aspirin inhibits platelet aggregation.", "The moon orbits Earth."],
        )
        if len(ranks) != 2 or {int(item.get("index", -1)) for item in ranks} != {0, 1}:
            raise RuntimeError("Cohere reranker warmup returned an invalid result set")
        if int(ranks[0].get("index", -1)) != 0:
            raise RuntimeError("Cohere reranker warmup did not rank the clinically relevant passage first")
        if float(ranks[0].get("score") or 0.0) <= float(ranks[1].get("score") or 0.0):
            raise RuntimeError("Cohere reranker warmup did not assign a strictly higher relevant-passage score")

        completion = await llm.call_llm(
            [{"role": "user", "content": f"Reply with exactly: {FOUNDRY_PROBE_MARKER}"}],
            max_tokens=32,
            temperature=0.0,
        )
        completion_text = str(completion.get("content") or "").strip()
        if completion_text != FOUNDRY_PROBE_MARKER:
            raise RuntimeError(
                f"Foundry GPT identity/behavior probe returned {completion_text!r}; "
                f"expected exactly {FOUNDRY_PROBE_MARKER!r}"
            )
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
        "foundry_probe_marker": FOUNDRY_PROBE_MARKER,
        "foundry_probe_exact": True,
        "embedding_model": expected_embedding,
        "embedding_dimension": expected_dim,
        "embedding_protocol": embedding.protocol,
        "embedding_auth_scheme": os.getenv("CLOUD_EMBEDDING_AUTH_SCHEME", "bearer"),
        "embedding_input_types_verified": ["query", "document"],
        "embedding_uri_host": embedding.base_url.split("//", 1)[-1].split("/", 1)[0],
        "reranker_model": expected_reranker,
        "reranker_auth_scheme": os.getenv("CLOUD_RERANKER_AUTH_SCHEME", "bearer"),
        "reranker_uri_host": reranker.base_url.split("//", 1)[-1].split("/", 1)[0],
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(MANIFEST_DIR / "foundry-services.json", payload)
    print(
        "[warmup] Azure Foundry verified: "
        f"LLM={foundry_model}, embed={expected_embedding}/{expected_dim} protocol={embedding.protocol}, "
        f"rerank={expected_reranker}"
    )
    return payload


def warm_nist(force: bool) -> dict[str, Any]:
    verified = download_nist(force=force)
    print(f"[warmup] NIST CDS 2016 files verified: {', '.join(verified)}")
    return {"complete": True, "sha256": verified, "completed_at": datetime.now(timezone.utc).isoformat()}


def warm_trec_eval() -> dict[str, Any]:
    path = ensure_trec_eval()
    print(f"[warmup] trec_eval ready: {path}")
    return {"complete": True, "path": str(path), "completed_at": datetime.now(timezone.utc).isoformat()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--with-local-llm", action="store_true")
    parser.add_argument("--skip-trec", action="store_true")
    parser.add_argument("--skip-nist", action="store_true")
    parser.add_argument("--skip-trec-eval", action="store_true")
    parser.add_argument("--skip-foundry", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--system-eval", action="store_true", help="Use the MedSwin 2.0 system-eval contract.")
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
    if not args.skip_nist:
        results["nist"] = warm_nist(args.force)
        verify_nist()
    if not args.skip_trec_eval:
        results["trec_eval"] = warm_trec_eval()
    need_local_llm = args.with_local_llm or args.system_eval or os.getenv("SYSTEM_EVAL_NEED_LOCAL_LLM") == "1"
    if need_local_llm and not args.skip_model:
        results["model"] = warm_model(args.model_id, Path(args.model_path), args.force)
    if args.system_eval and not args.skip_model:
        results["nli"] = warm_nli(args.force)
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
