#!/usr/bin/env python3
"""Smoke-test the local MedEmbed and BGE reranker weights.

The cases are public medical sentences. They are not a clinical validation.
The script writes nothing except model files under ./models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED_REPO = "abhinand/MedEmbed-large-v0.1"
RERANK_REPO = "BAAI/bge-reranker-v2-m3"
EMBED_DIR = ROOT / "models" / "MedEmbed-large-v0.1"
RERANK_DIR = ROOT / "models" / "bge-reranker-v2-m3"

CASES = [
    {
        "query": "first line treatment for community acquired pneumonia in adults",
        "relevant": "Amoxicillin is recommended for low-severity community acquired pneumonia in adults.",
        "irrelevant": "The moon completes one orbit of Earth in about 27 days.",
    },
    {
        "query": "metformin contraindication when renal function is severely reduced",
        "relevant": "Do not start metformin when the eGFR is below 30 mL/min/1.73 m2.",
        "irrelevant": "Stainless steel resists corrosion because of a chromium oxide film.",
    },
    {
        "query": "aspirin effect on platelet aggregation",
        "relevant": "Aspirin inhibits platelet cyclooxygenase and reduces platelet aggregation.",
        "irrelevant": "A violin string vibrates at a frequency set by its length and tension.",
    },
]


def _token() -> str | None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return os.environ.get("HF_TOKEN")


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    token = _token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    snapshot_download(EMBED_REPO, local_dir=str(EMBED_DIR), token=token)
    snapshot_download(RERANK_REPO, local_dir=str(RERANK_DIR), token=token)
    device = _device()
    embedder = SentenceTransformer(str(EMBED_DIR), device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(RERANK_DIR))
    reranker = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR))
    reranker.to(device)
    reranker.eval()

    dimension = int(embedder.get_sentence_embedding_dimension())
    rows = []
    for case in CASES:
        vectors = embedder.encode(
            [case["query"], case["relevant"], case["irrelevant"]],
            normalize_embeddings=True,
        )
        relevant_cos = float(vectors[0] @ vectors[1])
        irrelevant_cos = float(vectors[0] @ vectors[2])
        pairs = tokenizer(
            [case["query"], case["query"]],
            [case["relevant"], case["irrelevant"]],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        pairs = {key: value.to(device) for key, value in pairs.items()}
        with torch.no_grad():
            logits = reranker(**pairs).logits.view(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
        rows.append(
            {
                "query": case["query"],
                "relevant_cosine": round(relevant_cos, 4),
                "irrelevant_cosine": round(irrelevant_cos, 4),
                "relevant_rerank": round(float(scores[0]), 4),
                "irrelevant_rerank": round(float(scores[1]), 4),
                "cosine_orders_relevant_first": relevant_cos > irrelevant_cos,
                "rerank_orders_relevant_first": scores[0] > scores[1],
            }
        )

    passed = dimension == 1024 and all(
        row["cosine_orders_relevant_first"] and row["rerank_orders_relevant_first"] for row in rows
    )
    report = {
        "device": device,
        "embed_repo": EMBED_REPO,
        "rerank_repo": RERANK_REPO,
        "embedding_dimension": dimension,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "cases": rows,
        "passed": passed,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
