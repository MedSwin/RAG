"""T3 rating schemas. Task A and Task B are separate. No Med-PaLM axes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ActionLabel = Literal["answer", "abstain"]
SupportLabel = Literal["supported", "unsupported", "contradictory"]


class TaskAItem(BaseModel):
    item_id: str
    topic_id: int
    pipeline: Literal["medswin", "naive"]
    generator: Literal["cloud", "medswin"]
    topic_type: str
    note: str
    type_question: str
    packed_snippets: list[dict] = Field(default_factory=list)
    gold_action: ActionLabel | None = None
    rater_a: ActionLabel | None = None
    rater_b: ActionLabel | None = None
    adjudicated: ActionLabel | None = None


class TaskBItem(BaseModel):
    item_id: str
    topic_id: int
    pipeline: Literal["medswin", "naive"]
    generator: Literal["cloud", "medswin"]
    claim_id: int
    claim: str
    snippets: list[dict] = Field(default_factory=list)
    uncited: bool = False
    rater_a: SupportLabel | None = None
    rater_b: SupportLabel | None = None
    adjudicated: SupportLabel | None = None


class Confusion2x2(BaseModel):
    true_answer: int = 0
    false_answer: int = 0
    false_abstain: int = 0
    true_abstain: int = 0

    @property
    def n(self) -> int:
        return self.true_answer + self.false_answer + self.false_abstain + self.true_abstain

    def rates(self) -> dict[str, float]:
        n = max(self.n, 1)
        n_s = max(self.false_answer + self.true_abstain, 1)
        n_a = max(self.true_answer + self.false_abstain, 1)
        return {
            "accuracy": (self.true_answer + self.true_abstain) / n,
            "false_answer_over_n": self.false_answer / n,
            "false_answer_over_gold_s": self.false_answer / n_s,
            "false_abstain_over_n": self.false_abstain / n,
            "false_abstain_over_gold_a": self.false_abstain / n_a,
            "sensitivity": self.true_answer / n_a,
            "specificity": self.true_abstain / n_s,
        }


def score_action(system_answered: bool, gold: ActionLabel) -> str:
    if system_answered and gold == "answer":
        return "true_answer"
    if system_answered and gold == "abstain":
        return "false_answer"
    if (not system_answered) and gold == "answer":
        return "false_abstain"
    return "true_abstain"
