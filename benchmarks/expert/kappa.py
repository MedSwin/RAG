"""Cohen and Randolph kappa with a normal-approximation 95% CI."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def _pairs(left: Sequence[str], right: Sequence[str]) -> list[tuple[str, str]]:
    if len(left) != len(right):
        raise ValueError("rater vectors must be the same length")
    return list(zip(left, right))


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> dict[str, float]:
    pairs = _pairs(left, right)
    n = len(pairs)
    if n == 0:
        return {"kappa": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0.0}
    agree = sum(1 for a, b in pairs if a == b) / n
    left_counts = Counter(a for a, _b in pairs)
    right_counts = Counter(b for _a, b in pairs)
    chance = sum((left_counts[label] / n) * (right_counts[label] / n) for label in set(left_counts) | set(right_counts))
    denom = 1.0 - chance
    kappa = 0.0 if denom == 0 else (agree - chance) / denom
    se = math.sqrt(max(agree * (1 - agree) / n, 0.0)) / max(denom, 1e-9)
    return {
        "kappa": kappa,
        "ci95_lo": kappa - 1.96 * se,
        "ci95_hi": kappa + 1.96 * se,
        "n": float(n),
        "po": agree,
        "pe": chance,
    }


def randolph_kappa(
    left: Sequence[str],
    right: Sequence[str],
    n_categories: int | None = None,
) -> dict[str, float]:
    pairs = _pairs(left, right)
    n = len(pairs)
    labels = sorted(set(left) | set(right))
    k = max(n_categories or len(labels), 1)
    if n == 0:
        return {"kappa": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n": 0.0}
    agree = sum(1 for a, b in pairs if a == b) / n
    chance = 1.0 / k
    denom = 1.0 - chance
    kappa = 0.0 if denom == 0 else (agree - chance) / denom
    se = math.sqrt(max(agree * (1 - agree) / n, 0.0)) / max(denom, 1e-9)
    return {
        "kappa": kappa,
        "ci95_lo": kappa - 1.96 * se,
        "ci95_hi": kappa + 1.96 * se,
        "n": float(n),
        "po": agree,
        "pe": chance,
    }
