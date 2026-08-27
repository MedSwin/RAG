import subprocess
from pathlib import Path


def test_start_local_usage_lists_paper_eval_examples():
    text = Path("scripts/start-local.sh").read_text(encoding="utf-8")
    assert "./scripts/start-local.sh paper-eval" in text
    assert "--systems bm25,dense" in text
    assert "--stage emit --systems cascade" in text
    assert "--pipeline both --generator cloud --stage t3" in text
    assert "--pipeline medswin --stage t4" in text
    assert "medswin-mongodb" in text
    assert "mongo:7.0" in text
    assert "rag_mongodb" not in text
    assert "EVAL_WARMUP_ON_START:-false" in text
    assert "ensure_eval" not in text
    assert "EVAL_PORT" not in text


def test_warmup_downloads_nist_and_skips_hf_by_default():
    text = Path("scripts/warmup-eval.py").read_text(encoding="utf-8")
    assert "warm_nist" in text
    assert "ensure_trec_eval" in text
    assert "PAPER_EVAL_NEED_LOCAL_LLM" in text
    assert "EXPECTED_TREC_DOCS = EXPECTED_DOCUMENTS" in text


def test_illegal_paper_eval_combos_exit_before_bootstrap():
    text = Path("scripts/start-local.sh").read_text(encoding="utf-8")
    reject_at = text.index("reject_illegal_paper_eval()")
    venv_at = text.index('source "${VENV}/bin/activate"')
    assert reject_at < venv_at
    cases = [
        ["paper-eval", "--stage", "t3", "--systems", "bm25"],
        ["paper-eval", "--stage", "emit", "--pipeline", "both"],
        ["paper-eval", "--stage", "t3", "--generator", "medswin"],
    ]
    for args in cases:
        result = subprocess.run(
            ["bash", "scripts/start-local.sh", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1, args
        assert "paper-eval" in (result.stdout + result.stderr).lower() or "T1" in (result.stdout + result.stderr) or "Foundry" in (result.stdout + result.stderr)
