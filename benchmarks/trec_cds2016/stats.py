"""Pre-registered T1 statistics: paired t / randomization, bootstrap CI, BH-FDR."""

from __future__ import annotations

import math
import random
from typing import Iterable

from .contract import TOPIC_IDS

PREGISTERED_CONTRASTS = (
    ("bm25", "rrf"),
    ("dense", "rrf"),
    ("rrf", "cascade"),
)


def paired_differences(left: dict[str, float], right: dict[str, float]) -> list[float]:
    return [right[str(qid)] - left[str(qid)] for qid in TOPIC_IDS]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def paired_t_pvalue(deltas: list[float]) -> float:
    n = len(deltas)
    if n < 2:
        return 1.0
    mu = mean(deltas)
    variance = sum((item - mu) ** 2 for item in deltas) / (n - 1)
    if variance <= 0:
        return 0.0 if mu != 0 else 1.0
    t_stat = mu / math.sqrt(variance / n)
    # Regularized incomplete beta survival for two-sided Student-t.
    df = n - 1
    x = df / (df + t_stat * t_stat)
    return _betainc_reg(0.5 * df, 0.5, x)


def _betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 1.0
    if x >= 1:
        return 0.0
    try:
        from math import lgamma

        ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
        term = math.exp(a * math.log(x) + b * math.log(1 - x) - ln_beta) / a
        total = term
        for n in range(1, 200):
            term *= (a + n - 1) * x / (a + n)
            total += term
            if abs(term) < 1e-12:
                break
        cdf = total
        # I_x(a,b) approximated; two-sided p ≈ I.
        return min(1.0, max(0.0, cdf))
    except Exception:
        return 1.0


def randomization_pvalue(deltas: list[float], draws: int = 10000, seed: int = 1337) -> float:
    observed = abs(mean(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        sample = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(mean(sample)) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def bootstrap_mean_ci(values: list[float], draws: int = 10000, seed: int = 1337, alpha: float = 0.05) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * draws)]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return mean(values), lo, hi


def benjamini_hochberg(pvalues: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
    ranked = sorted(pvalues, key=lambda item: item[1])
    m = len(ranked)
    adjusted: list[float] = [0.0] * m
    running = 1.0
    for index in range(m - 1, -1, -1):
        _name, pvalue = ranked[index]
        running = min(running, pvalue * m / (index + 1))
        adjusted[index] = min(1.0, running)
    return [
        (name, pvalue, adj, adj < 0.05)
        for (name, pvalue), adj in zip(ranked, adjusted)
    ]


def contrast_infndcg(
    per_system: dict[str, dict[str, float]],
) -> dict[str, object]:
    family: list[tuple[str, float]] = []
    details: dict[str, object] = {}
    for left, right in PREGISTERED_CONTRASTS:
        deltas = paired_differences(per_system[left], per_system[right])
        t_p = paired_t_pvalue(deltas)
        rand_p = randomization_pvalue(deltas)
        mu, lo, hi = bootstrap_mean_ci(deltas)
        name = f"{left}_vs_{right}"
        family.append((name, rand_p))
        details[name] = {
            "mean_delta": mu,
            "ci95": [lo, hi],
            "paired_t_p": t_p,
            "randomization_p": rand_p,
        }
    details["fdr"] = [
        {"contrast": name, "p": pvalue, "q": qvalue, "star": star}
        for name, pvalue, qvalue, star in benjamini_hochberg(family)
    ]
    return details
