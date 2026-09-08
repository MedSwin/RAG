"""Pre-registered paired bootstrap, randomization, and Holm correction."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Iterable, Sequence

import numpy as np


def paired_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired inputs must have the same non-zero length")
    return float(np.mean(np.asarray(right, dtype=float) - np.asarray(left, dtype=float)))


def bootstrap_ci(deltas: Sequence[float], *, draws: int = 10_000, seed: str = "system-eval") -> tuple[float, float]:
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return (math.nan, math.nan)
    rng = np.random.default_rng(int(sha256(seed.encode()).hexdigest()[:16], 16))
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def sign_randomization_pvalue(deltas: Sequence[float], *, draws: int = 10_000, seed: str = "system-eval") -> float:
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return math.nan
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(int(sha256((seed + "-sign").encode()).hexdigest()[:16], 16))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (draws + 1))


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    raw = list(pvalues)
    ordered = sorted(enumerate(raw), key=lambda item: item[1])
    adjusted = [math.nan] * len(raw)
    running = 0.0
    count = len(raw)
    for index, (original, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[original] = running
    return adjusted


def verdict(*, retrieval_delta: float, answer_delta: float, retrieval_ci: tuple[float, float], answer_ci: tuple[float, float], retrieval_p: float, answer_p: float, coverage: float) -> str:
    if coverage < 0.90:
        return "insufficient reliability"
    retrieval = retrieval_delta > 0 and retrieval_ci[0] > 0 and retrieval_p < 0.05
    answer = answer_delta > 0 and answer_ci[0] > 0 and answer_p < 0.05
    if retrieval and answer:
        return "overall improvement"
    if retrieval:
        return "retrieval-only improvement"
    if answer:
        return "answer-only improvement"
    return "no demonstrated improvement"
