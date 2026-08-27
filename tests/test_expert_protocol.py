import subprocess
import sys

from benchmarks.expert.claims import CLAIM_CAP, attach_citations, split_claims
from benchmarks.expert.kappa import cohen_kappa, randolph_kappa
from benchmarks.expert.schema import Confusion2x2, score_action
from benchmarks.trec_cds2016.topics import CdsTopic


def test_claim_splitter_caps_and_splits_conjuncts():
    text = (
        "The patient has pneumonia and the patient should receive ceftriaxone. "
        "Blood cultures are indicated. Imaging confirms a left-lower-lobe infiltrate. "
        "Supportive oxygen is reasonable. Steroids are not first-line here. "
        "Duration is five days. Follow-up is needed. Extra claim nine."
    )
    claims = split_claims(text, cap=CLAIM_CAP)
    assert 1 <= len(claims) <= CLAIM_CAP
    attached = attach_citations(claims, [{"doc_id": "1", "chunk_id": "c1", "text": "snippet"}])
    assert attached[0]["uncited"] is False
    assert attach_citations(["uncited claim"], [])[0]["uncited"] is True


def test_kappa_perfect_and_chance():
    perfect = cohen_kappa(["answer", "abstain"], ["answer", "abstain"])
    assert perfect["kappa"] == 1.0
    randolph = randolph_kappa(["answer", "answer"], ["answer", "answer"], n_categories=2)
    assert randolph["kappa"] == 1.0


def test_naive_two_by_two_specificity_is_zero():
    matrix = Confusion2x2(true_answer=20, false_answer=10, false_abstain=0, true_abstain=0)
    rates = matrix.rates()
    assert rates["specificity"] == 0.0
    assert score_action(True, "abstain") == "false_answer"
    assert score_action(False, "answer") == "false_abstain"


def test_t3_query_is_type_question_only():
    topic = CdsTopic(1, "treatment", "long note", "description", "summary")
    assert topic.t3_query() == "How should the patient be treated?"
    assert topic.patient_id == "trec-cds-1"


def test_t4_fails_closed_without_packs(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.expert.t4_automatic", "--packs-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "T4 packs missing" in result.stderr + result.stdout
