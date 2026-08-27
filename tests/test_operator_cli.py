from argparse import Namespace

from app.cli.operator import _looks_like_question, _parser
from app.cli.surfaces import format_portals, format_status, portal_urls


def test_portal_urls_cover_clinician_dashboard_and_docs():
    urls = portal_urls("http://127.0.0.1:8100")
    assert urls["clinician"].endswith("/app/")
    assert urls["dashboard"].endswith("/api/v1/dashboard/")
    assert urls["docs"].endswith("/docs")
    assert "eval" not in urls


def test_format_portals_lists_operator_surfaces():
    text = format_portals("http://127.0.0.1:8100")
    assert "Clinician CDS" in text
    assert "Ops dashboard" in text
    assert "benchmark UI" not in text
    assert "8200" not in text


def test_format_status_flags_missing_embeddings_and_index():
    text = format_status(
        {
            "api_base": "http://127.0.0.1:8100",
            "org_id": "demo-org",
            "api": {"ok": True, "body": {"cloud_mode": False, "embedding_model": "loaded", "reranker_model": "not_loaded"}},
            "naive_ready": {"ok": True, "body": {"mongo": True, "chunk_count": 12, "embedded_count": 0, "index_exists": False}},
            "storage": {
                "ok": True,
                "body": {
                    "total_chunks": 12,
                    "active_embeddings": 0,
                    "source_counts": {"CPG": 0, "EMR": 0, "LIT": 12, "SAFETY": 0},
                    "index_exists": False,
                    "index_provenance_valid": False,
                },
            },
            "local_llm": {"ok": False},
            "paper_eval": {"nist_complete": False, "nist_files": {}, "runs": [], "scores": []},
        }
    )
    assert "chunks exist but 0 embeddings" in text
    assert "ANN index is missing" in text
    assert "paper-eval" in text
    assert "Eval portal" not in text


def test_operator_parser_has_no_eval_command():
    args = _parser().parse_args(["status"])
    assert args.command == "status"
    choices = _parser()._option_string_actions
    assert "--eval-action" not in choices
    assert "--with-eval" not in choices


def test_looks_like_question_accepts_clinical_text():
    assert _looks_like_question("Can this patient continue metformin?")
    assert not _looks_like_question("status")
    assert not _looks_like_question("1")


def test_prompt_args_namespace_has_run_once_fields():
    from app.cli.operator import _prompt_args

    args = _prompt_args(
        Namespace(
            base_url="http://127.0.0.1:8100",
            mode="both",
            question="q",
            org_id="demo-org",
            user_id="clinician-1",
            patient_id="",
            session_id="",
            top_k=5,
            timeout=30.0,
            json=False,
        )
    )
    assert args.mode == "both"
    assert args.top_k == 5
