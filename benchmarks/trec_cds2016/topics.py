"""T1 vs T3 query construction. ir_datasets default_text() is description — never use it."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .contract import EXPECTED_QUERIES, TYPE_QUESTIONS
from .nist import nist_paths

TopicField = Literal["note", "summary"]


@dataclass(frozen=True)
class CdsTopic:
    number: int
    topic_type: str
    note: str
    description: str
    summary: str

    @property
    def type_question(self) -> str:
        return TYPE_QUESTIONS[self.topic_type]

    def t1_query(self, field: TopicField = "note") -> str:
        body = self.note if field == "note" else self.summary
        return f"{body.strip()}\n\n{self.type_question}"

    def t3_query(self) -> str:
        return self.type_question

    @property
    def patient_id(self) -> str:
        return f"trec-cds-{self.number}"


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def load_topics(topics_path: Path | None = None) -> list[CdsTopic]:
    path = topics_path or nist_paths()["topics2016.xml"]
    root = ET.parse(path).getroot()
    topics: list[CdsTopic] = []
    for node in root.findall("topic"):
        number = int(node.attrib["number"])
        topic_type = str(node.attrib["type"]).strip().lower()
        if topic_type not in TYPE_QUESTIONS:
            raise RuntimeError(f"Unknown topic type {topic_type!r} on topic {number}")
        topics.append(
            CdsTopic(
                number=number,
                topic_type=topic_type,
                note=_text(node.find("note")),
                description=_text(node.find("description")),
                summary=_text(node.find("summary")),
            )
        )
    topics.sort(key=lambda item: item.number)
    if len(topics) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} topics, found {len(topics)}")
    if [item.number for item in topics] != list(range(1, EXPECTED_QUERIES + 1)):
        raise RuntimeError("Topic numbers are not 1-30")
    return topics


def iter_t1_queries(field: TopicField = "note", topics: Iterable[CdsTopic] | None = None) -> list[tuple[int, str]]:
    return [(topic.number, topic.t1_query(field)) for topic in (topics or load_topics())]
