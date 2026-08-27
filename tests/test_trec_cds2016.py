from pathlib import Path

import pytest

from benchmarks.trec_cds2016.contract import (
    CASCADE_RERANK,
    EXPECTED_DOCUMENTS,
    EXPECTED_QRELS,
    RRF_K,
    RRF_K_BM25,
    RRF_K_DENSE,
    RUN_DEPTH,
    TYPE_QUESTIONS,
)
from benchmarks.trec_cds2016.nist import NIST_FILES
from benchmarks.trec_cds2016.ranking import cascade_fill, max_pool_pmcids, rrf_combine
from benchmarks.trec_cds2016.stats import PREGISTERED_CONTRASTS, benjamini_hochberg, contrast_infndcg
from benchmarks.trec_cds2016.topics import CdsTopic
from benchmarks.trec_cds2016.validate import validate_run


def test_contract_constants_are_official():
    assert EXPECTED_DOCUMENTS == 1_255_260
    assert EXPECTED_QRELS == 37_707
    assert RRF_K == 60
    assert RRF_K_BM25 == 4000
    assert RRF_K_DENSE == 4000
    assert CASCADE_RERANK == 300
    assert RUN_DEPTH == 1000
    assert TYPE_QUESTIONS["diagnosis"] == "What is the patient's diagnosis?"
    assert TYPE_QUESTIONS["test"] == "What tests should the patient receive?"
    assert TYPE_QUESTIONS["treatment"] == "How should the patient be treated?"


def test_nist_pins_official_urls_and_sha256():
    assert NIST_FILES["topics2016.xml"]["url"] == "https://trec.nist.gov/data/clinical/topics2016.xml"
    assert NIST_FILES["qrels-treceval-2016.txt"]["url"] == "https://trec.nist.gov/data/clinical/qrels-treceval-2016.txt"
    assert NIST_FILES["qrels-sampleval-2016.txt"]["url"] == "https://trec.nist.gov/data/clinical/qrels-sampleval-2016.txt"
    assert NIST_FILES["sample_eval.pl"]["url"] == "https://trec.nist.gov/data/clinical/sample_eval.pl"
    assert len(NIST_FILES["qrels-sampleval-2016.txt"]["sha256"]) == 64
    assert NIST_FILES["qrels-treceval-2016.txt"]["sha256"] != NIST_FILES["qrels-sampleval-2016.txt"]["sha256"]


def test_t1_and_t3_queries_are_split():
    topic = CdsTopic(
        number=1,
        topic_type="diagnosis",
        note="ICU note about fever.",
        description="A 40 year old man with fever.",
        summary="Adult with fever.",
    )
    assert topic.t1_query("note").startswith("ICU note")
    assert "What is the patient's diagnosis?" in topic.t1_query("note")
    assert topic.t3_query() == "What is the patient's diagnosis?"
    assert topic.description not in topic.t1_query("note")
    assert topic.note not in topic.t3_query()


def test_max_pool_and_rrf_and_cascade():
    pooled = max_pool_pmcids([("PMC3148967", 0.2), ("3148967", 0.9), ("bad", 0.5)])
    assert pooled[0] == ("3148967", 0.9)
    rrf = rrf_combine([["1", "2"], ["2", "3"]], k=60, depth=10)
    assert [doc for doc, _score in rrf][0] == "2"
    filled = cascade_fill([("9", 1.0), ("8", 0.5)], [("1", 0.1), ("9", 0.2), ("7", 0.05)], head=2, depth=3)
    assert [doc for doc, _score in filled] == ["9", "8", "1"]


def test_run_validator_accepts_legal_file(tmp_path):
    lines = []
    for qid in range(1, 31):
        for rank in range(1, 4):
            score = 1.0 - rank * 0.1
            lines.append(f"{qid} Q0 {1000 + rank} {rank} {score:.4f} msbm25note")
    path = tmp_path / "msbm25note.run"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_run(path)
    assert report["run_name"] == "msbm25note"
    assert report["topics"] == 30


def test_run_validator_rejects_prefixed_pmcid(tmp_path):
    lines = [f"{qid} Q0 PMC3148967 1 0.9 badname!" for qid in range(1, 31)]
    path = tmp_path / "bad.run"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PMCID|RUN_NAME"):
        validate_run(path)


def test_preregistered_contrasts_are_the_only_starred_family():
    assert PREGISTERED_CONTRASTS == (("bm25", "rrf"), ("dense", "rrf"), ("rrf", "cascade"))
    per_system = {
        "bm25": {str(qid): 0.10 for qid in range(1, 31)},
        "dense": {str(qid): 0.11 for qid in range(1, 31)},
        "rrf": {str(qid): 0.20 for qid in range(1, 31)},
        "cascade": {str(qid): 0.21 for qid in range(1, 31)},
    }
    result = contrast_infndcg(per_system)
    assert {row["contrast"] for row in result["fdr"]} == {
        "bm25_vs_rrf",
        "dense_vs_rrf",
        "rrf_vs_cascade",
    }
    adjusted = benjamini_hochberg([("a", 0.01), ("b", 0.04), ("c", 0.20)])
    assert adjusted[0][3] is True


def test_prereg_file_pins_sha256_and_contrasts():
    text = Path("benchmarks/trec_cds2016/PREREG.md").read_text(encoding="utf-8")
    assert "167541d16ab0986fd36045fb4e1104fccb6c8df18e1842b7ee448e7639479767" in text
    assert "sample_eval.pl" in text
    assert "BM25 vs RRF" in text
    assert "RRF vs RRF+cascade" in text
