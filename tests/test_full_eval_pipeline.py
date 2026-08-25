from pathlib import Path

import pytest

from eval.app.full_contract import expected_matrix_keys, is_publication_matrix, resolve_pipelines

ROOT = Path(__file__).resolve().parents[1]


def test_full_matrix_pipeline_filter_derives_expected_keys():
    assert resolve_pipelines("both") == ("naive_rag", "medswin")
    assert resolve_pipelines("naive_rag") == ("naive_rag",)
    assert resolve_pipelines("medswin") == ("medswin",)

    naive_keys = expected_matrix_keys(resolve_pipelines("naive_rag"))
    assert naive_keys == {"naive_rag:medswin_local", "naive_rag:foundry"}
    assert "medswin:foundry" not in naive_keys
    assert "medswin:medswin_local" not in naive_keys

    assert expected_matrix_keys(resolve_pipelines("both")) == {
        "naive_rag:medswin_local",
        "naive_rag:foundry",
        "medswin:medswin_local",
        "medswin:foundry",
    }
    assert is_publication_matrix(resolve_pipelines("both")) is True
    assert is_publication_matrix(resolve_pipelines("naive_rag")) is False
    assert is_publication_matrix(resolve_pipelines("medswin")) is False
    with pytest.raises(ValueError, match="Unsupported full-eval pipeline"):
        resolve_pipelines("both_naive")


def test_start_local_forwards_pipeline_into_full_matrix():
    script = (ROOT / "scripts" / "start-local.sh").read_text(encoding="utf-8")
    assert 'matrix_args=(--org-id "$bench_org" --pipeline "$PIPELINE")' in script
    assert "eval/scripts/run_full_matrix.py" in script


def test_start_local_requires_python_312_or_313():
    script = (ROOT / "scripts" / "start-local.sh").read_text(encoding="utf-8")
    assert '[[ "$ver" == "3.12" || "$ver" == "3.13" ]]' in script
    assert 'candidates+=(python3.13 python3.12)' in script


def test_start_local_warmup_avoids_empty_array_expansion():
    script = (ROOT / "scripts" / "start-local.sh").read_text(encoding="utf-8")
    assert "run_eval_warmup()" in script
    assert 'CLOUD_MODE=true python3 scripts/warmup-eval.py --force' in script
    assert 'CLOUD_MODE=true python3 scripts/warmup-eval.py\n' in script or (
        'CLOUD_MODE=true python3 scripts/warmup-eval.py' in script
        and 'local args=()' not in script
    )
