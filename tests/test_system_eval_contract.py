from pathlib import Path

import pytest

from benchmarks.system_eval.contracts import expand_matrix, parse_generators, parse_pipelines
from benchmarks.system_eval.metrics import ndcg_at, precision_at, recall_at, rouge_su4
from benchmarks.system_eval.statistics import bootstrap_ci, holm_adjust, sign_randomization_pvalue, verdict
from app.api.v1.endpoints.pipelines import ChatConstraints
from app.medswin.orchestrator import MedSwinOrchestrator
from app.schemas.enums import PipelineId


def test_matrix_expands_exact_six_cells_and_september_exact_four():
    assert [(cell.pipeline, cell.generator) for cell in expand_matrix()] == [
        ("naive", "cloud"), ("two-stage", "cloud"), ("full", "cloud"),
        ("naive", "medswin"), ("two-stage", "medswin"), ("full", "medswin"),
    ]
    assert [(cell.pipeline, cell.generator) for cell in expand_matrix(custom="combination-september")] == [
        ("naive", "cloud"), ("two-stage", "cloud"), ("naive", "medswin"), ("two-stage", "medswin"),
    ]


@pytest.mark.parametrize("value", ["both", "medswin", "", "naive,full-system"])
def test_pipeline_selection_rejects_legacy_values(value):
    with pytest.raises(ValueError):
        parse_pipelines(value)


def test_generator_selection_and_ranking_metrics_are_fixed_cutoff():
    assert parse_generators("all") == ("cloud", "medswin")
    ranking, gains = ["a", "x", "b"], {"a": 2, "b": 1}
    assert precision_at(ranking, {"a", "b"}, 5) == 0.4
    assert recall_at(ranking, {"a", "b"}, 5) == 1.0
    assert 0 < ndcg_at(ranking, gains, 5) <= 1


def test_rouge_and_paired_statistics_are_deterministic():
    assert rouge_su4("aspirin helps", "aspirin helps") ["f1"] == 1.0
    values = [0.1, 0.2, 0.3]
    assert bootstrap_ci(values, draws=100, seed="x") == bootstrap_ci(values, draws=100, seed="x")
    assert 0 <= sign_randomization_pvalue(values, draws=100, seed="x") <= 1
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    assert verdict(retrieval_delta=.1, answer_delta=.1, retrieval_ci=(.01,.2), answer_ci=(.01,.2), retrieval_p=.01, answer_p=.01, coverage=.9) == "overall improvement"


def test_start_script_exposes_v2_and_rejects_old_command():
    script = Path("scripts/start-local.sh").read_text(encoding="utf-8")
    assert "system-eval" in script
    assert "combination-september" in script
    assert "--allow-local-t3" in script  # explicit migration rejection only


def test_public_constraints_reject_retired_ablation_flags():
    with pytest.raises(Exception):
        ChatConstraints.model_validate({"disable_mac": True})
    with pytest.raises(Exception):
        ChatConstraints.model_validate({"disable_gate": True})


def test_two_stage_is_a_typed_full_policy_without_mac_clients():
    two_stage = MedSwinOrchestrator(pipeline=PipelineId.TWO_STAGE)
    try:
        assert two_stage.policy.enable_gate is True
        assert two_stage.policy.enable_retrieve_more is True
        assert two_stage.policy.enable_mac is False
        assert two_stage.emr_agent is None
    finally:
        import asyncio
        asyncio.run(two_stage.close())
