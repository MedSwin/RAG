import json
from dataclasses import dataclass

import pytest

from eval.scripts.ingest_trec_pmc import iter_docs, load_gold_doc_ids


@dataclass
class _Doc:
    doc_id: str


class _Dataset:
    def __init__(self, docs):
        self._docs = docs

    def docs_iter(self):
        yield from self._docs


def test_load_gold_doc_ids_from_cases(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"gold_doc_ids": ["gold-1", "gold-2"]}) + "\n"
        + json.dumps({"gold_doc_ids": ["gold-2", "gold-3"]}) + "\n",
        encoding="utf-8",
    )

    assert load_gold_doc_ids(str(cases_path)) == {"gold-1", "gold-2", "gold-3"}


def test_iter_docs_yields_judged_pool_before_negatives():
    docs = [_Doc("neg-1"), _Doc("gold-1"), _Doc("neg-2"), _Doc("gold-2")]

    yielded = list(iter_docs(_Dataset(docs), limit=1, sample_size=1, seed=7, priority_doc_ids={"gold-1", "gold-2"}))

    assert [doc.doc_id for doc in yielded] == ["gold-1", "gold-2", "neg-1"]


def test_iter_docs_can_require_all_judged_docs():
    with pytest.raises(RuntimeError, match="judged qrel documents"):
        list(
            iter_docs(
                _Dataset([_Doc("gold-1")]),
                limit=None,
                sample_size=1,
                seed=7,
                priority_doc_ids={"gold-1", "missing"},
                require_priority_docs=True,
            )
        )
