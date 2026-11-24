"""MedEmbed pooling and the local inference device.

MedEmbed-large-v0.1 follows BGE-large: the sentence vector is the CLS token,
then L2 normalisation. Mean-pooling the token states is a different vector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def inference_device() -> torch.device:
    """Use CUDA, then Apple Metal, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cls_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return one L2-normalised CLS vector per row.

    ``attention_mask`` is accepted so callers can keep the same signature as
    the old mean-pool helper. CLS does not use the mask.
    """
    del attention_mask
    return F.normalize(token_embeddings[:, 0], p=2, dim=1)
