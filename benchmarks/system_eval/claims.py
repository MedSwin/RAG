"""Versioned, non-clinician claim diagnostics backed by pinned local BioMedNLI."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Iterable

CLAIM_SPLITTER_VERSION = "sentence-conjunction-v1"
NLI_MODEL = "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"
NLI_REVISION = "e9c646143e4b88035c7a5505dc4dfaba59656bfd"
_SPLIT = re.compile(r"(?:[.!?]+\s+|\s+(?:and|but|however|although)\s+)", re.I)


def split_claims(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(str(text or "")) if len(part.strip().split()) >= 3]


@dataclass(frozen=True)
class NliResult:
    label: str
    probabilities: dict[str, float]


class LocalNliJudge:
    """Lazy immutable local model wrapper; tested generators are never judges."""

    def __init__(self, model: str = NLI_MODEL, revision: str = NLI_REVISION):
        local_path = os.getenv("SYSTEM_EVAL_NLI_PATH", "models/system-eval-nli")
        self.model = local_path if os.path.isdir(local_path) else model
        self.model_id = model
        self.revision = revision
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, revision=None if self.model != self.model_id else self.revision, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model, revision=None if self.model != self.model_id else self.revision, local_files_only=True)
        self._model.eval()

    def judge(self, premise: str, hypothesis: str) -> NliResult:
        self._load()
        import torch
        inputs = self._tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labels = {str(value).lower(): index for index, value in self._model.config.id2label.items()}
        canonical = {
            "entailment": float(probabilities[labels["entailment"]]),
            "neutral": float(probabilities[labels["neutral"]]),
            "contradiction": float(probabilities[labels["contradiction"]]),
        }
        return NliResult(max(canonical, key=canonical.get), canonical)


def claim_metrics(system_claims: Iterable[str], gold_claims: Iterable[str], cited_evidence: Iterable[str], judge: LocalNliJudge) -> dict[str, object]:
    system, gold, evidence = list(system_claims), list(gold_claims), list(cited_evidence)
    system_labels = [[judge.judge(reference, claim) for reference in gold] for claim in system]
    gold_labels = [[judge.judge(claim, generated) for generated in system] for claim in gold]
    supported_system = sum(any(item.label == "entailment" for item in rows) for rows in system_labels)
    supported_gold = sum(any(item.label == "entailment" for item in rows) for rows in gold_labels)
    precision = supported_system / len(system) if system else 0.0
    recall = supported_gold / len(gold) if gold else 0.0
    faithfulness = sum(any(judge.judge(passage, claim).label == "entailment" for passage in evidence) for claim in system) / len(system) if system else 0.0
    contradictions = sum(any(item.label == "contradiction" for item in rows) for rows in system_labels) / len(system) if system else 0.0
    return {
        "claim_precision": precision,
        "claim_recall": recall,
        "claim_f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "faithfulness": faithfulness,
        "contradiction_rate": contradictions,
        "unsupported_claim_rate": 1.0 - faithfulness,
        "raw_system_to_gold": [[result.probabilities for result in rows] for rows in system_labels],
        "raw_gold_to_system": [[result.probabilities for result in rows] for rows in gold_labels],
    }
