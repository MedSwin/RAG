#!/usr/bin/env python3
"""Load MedSwin-DaRE-TIES-KD-0.7 and generate a few tokens.

This is a load check. It is not a clinical validation.
Weights stay under ./models and are not committed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
LOCAL = ROOT / "models" / "MedSwin-DaRE-TIES-KD-0.7"
PROMPT = "Reply with the single word ready."
MAX_NEW_TOKENS = 8


def _token() -> str | None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
    return os.environ.get("HF_TOKEN")


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    token = _token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    snapshot_download(REPO, local_dir=str(LOCAL), token=token)
    downloaded = time.perf_counter()
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL))
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    loaded = time.perf_counter()
    encoded = tokenizer(PROMPT, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_tokens = output.shape[-1] - encoded["input_ids"].shape[-1]
    text = tokenizer.decode(output[0, -new_tokens:], skip_special_tokens=True)
    finished = time.perf_counter()
    report = {
        "repo": REPO,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "context": int(getattr(model.config, "max_position_embeddings", 0) or 0),
        "new_tokens": int(new_tokens),
        "output_chars": len(text.strip()),
        "output_preview": " ".join(text.split())[:80],
        "download_s": round(downloaded - started, 2),
        "load_s": round(loaded - downloaded, 2),
        "generate_s": round(finished - loaded, 2),
    }
    print(json.dumps(report))
    if report["parameters"] < 6_000_000_000 or report["new_tokens"] < 1 or report["output_chars"] < 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
