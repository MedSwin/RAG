"""Metric implementations shared by every system-eval pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

CUTOFFS = (5, 10, 20)


def _at(values: Sequence[str], cutoff: int) -> list[str]:
    return list(values[:cutoff])


def precision_at(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    return sum(doc in relevant for doc in _at(ranking, cutoff)) / cutoff


def recall_at(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    return 0.0 if not relevant else sum(doc in relevant for doc in _at(ranking, cutoff)) / len(relevant)


def ndcg_at(ranking: Sequence[str], gains: Mapping[str, int], cutoff: int) -> float:
    def dcg(items: Sequence[str]) -> float:
        return sum(((2 ** int(gains.get(doc, 0))) - 1) / math.log2(rank + 1) for rank, doc in enumerate(items, 1))

    actual = dcg(_at(ranking, cutoff))
    ideal_gains = sorted((int(value) for value in gains.values()), reverse=True)[:cutoff]
    ideal = sum(((2**gain) - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    return 0.0 if ideal == 0 else actual / ideal


def mrr_at(ranking: Sequence[str], relevant: set[str], cutoff: int = 20) -> float:
    for rank, doc_id in enumerate(_at(ranking, cutoff), 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ranking_metrics(ranking: Sequence[str], gains: Mapping[str, int]) -> dict[str, float]:
    relevant = {doc for doc, gain in gains.items() if gain > 0}
    result = {"mrr@20": mrr_at(ranking, relevant)}
    for cutoff in CUTOFFS:
        result[f"precision@{cutoff}"] = precision_at(ranking, relevant, cutoff)
        result[f"recall@{cutoff}"] = recall_at(ranking, relevant, cutoff)
        result[f"ndcg@{cutoff}"] = ndcg_at(ranking, gains, cutoff)
        result[f"candidate_shortfall@{cutoff}"] = float(max(0, cutoff - len(ranking)))
    return result


def unjudged_rate_at(ranking: Sequence[str], judged: set[str], cutoff: int) -> float:
    """TREC pooled-judgment diagnostic; unjudged is not treated as nonrelevant."""
    return sum(doc not in judged for doc in _at(ranking, cutoff)) / cutoff


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: Sequence[str], width: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + width]) for index in range(max(0, len(tokens) - width + 1)))


def rouge_n(system: str, reference: str, width: int = 2) -> dict[str, float]:
    system_grams, reference_grams = _ngrams(_tokens(system), width), _ngrams(_tokens(reference), width)
    overlap = sum((system_grams & reference_grams).values())
    precision = overlap / sum(system_grams.values()) if system_grams else 0.0
    recall = overlap / sum(reference_grams.values()) if reference_grams else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def rouge_su4(system: str, reference: str) -> dict[str, float]:
    """ROUGE-SU4-compatible skip-bigram plus unigram approximation.

    It is deterministic, shared across cells, and deliberately reports its
    implementation version in the run manifest rather than calling a hidden
    external scorer with different tokenisation.
    """
    def features(text: str) -> Counter[tuple[str, ...]]:
        words = _tokens(text)
        out: Counter[tuple[str, ...]] = Counter((word,) for word in words)
        for left in range(len(words)):
            for right in range(left + 1, min(len(words), left + 5)):
                out[(words[left], words[right])] += 1
        return out
    actual, ideal = features(system), features(reference)
    overlap = sum((actual & ideal).values())
    precision = overlap / sum(actual.values()) if actual else 0.0
    recall = overlap / sum(ideal.values()) if ideal else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def truncate_words(text: str, limit: int = 200) -> str:
    return " ".join(str(text or "").split()[:limit])


def binary_scores(predicted: bool, expected: bool) -> dict[str, float]:
    return {"accuracy": float(predicted == expected), "tp": float(predicted and expected), "fp": float(predicted and not expected), "fn": float(not predicted and expected)}


def list_scores(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    pred, gold = {str(item).lower().strip() for item in predicted}, {str(item).lower().strip() for item in expected}
    precision = len(pred & gold) / len(pred) if pred else 0.0
    recall = len(pred & gold) / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def normalise_exact(value: object) -> str:
    return " ".join(_tokens(str(value or "")))


def factoid_scores(predicted: object, expected: object) -> dict[str, float]:
    """BioASQ strict/lenient/MRR scoring for the normalized exact-answer form."""
    prediction = normalise_exact(predicted)
    answers = expected if isinstance(expected, list) else [expected]
    gold = [normalise_exact(item) for item in answers]
    strict = float(bool(gold) and prediction == gold[0])
    try:
        rank = gold.index(prediction) + 1
    except ValueError:
        rank = 0
    return {"strict_accuracy": strict, "lenient_accuracy": float(rank > 0), "mrr": 1.0 / rank if rank else 0.0}


def yesno_prediction(value: object) -> bool | None:
    normalized = normalise_exact(value)
    if normalized in {"yes", "true"}:
        return True
    if normalized in {"no", "false"}:
        return False
    return None
