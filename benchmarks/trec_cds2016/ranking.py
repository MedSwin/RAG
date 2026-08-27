"""Document ranking helpers: max-pool chunks to PMCID and RRF (Cormack 2009)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .contract import RRF_K, RUN_DEPTH


def bare_pmcid(doc_id: str | None) -> str | None:
    if doc_id is None:
        return None
    text = str(doc_id).strip()
    if text.lower().startswith("pmc"):
        text = text[3:]
    text = text.strip()
    if not text.isdigit():
        return None
    return str(int(text))


def max_pool_pmcids(
    items: Iterable[tuple[str | None, float]],
    depth: int = RUN_DEPTH,
) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for doc_id, score in items:
        pmcid = bare_pmcid(doc_id)
        if pmcid is None:
            continue
        current = best.get(pmcid)
        if current is None or float(score) > current:
            best[pmcid] = float(score)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:depth]


def rrf_combine(
    rankings: list[list[str]],
    k: int = RRF_K,
    depth: int = RUN_DEPTH,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            pmcid = bare_pmcid(doc_id)
            if pmcid is None or pmcid in seen:
                continue
            seen.add(pmcid)
            scores[pmcid] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:depth]


def cascade_fill(
    reranked: list[tuple[str, float]],
    tail: list[tuple[str, float]],
    head: int,
    depth: int = RUN_DEPTH,
) -> list[tuple[str, float]]:
    """Keep the reranked head order; fill remaining ranks from the unreranked tail."""
    ordered: list[tuple[str, float]] = []
    seen: set[str] = set()
    for doc_id, score in reranked[:head]:
        pmcid = bare_pmcid(doc_id)
        if pmcid is None or pmcid in seen:
            continue
        seen.add(pmcid)
        ordered.append((pmcid, float(score)))
    for doc_id, score in tail:
        if len(ordered) >= depth:
            break
        pmcid = bare_pmcid(doc_id)
        if pmcid is None or pmcid in seen:
            continue
        seen.add(pmcid)
        ordered.append((pmcid, float(score)))
    return ordered[:depth]
