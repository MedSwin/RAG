"""MedSwin local operator: ask, compare, evaluate, and open portals."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.cli.prompt import MODES, run_once
from app.cli.surfaces import (
    DEFAULT_API,
    DEFAULT_EVAL,
    ROOT,
    collect_status,
    ensure_api,
    ensure_eval,
    format_portals,
    format_status,
    http_json,
    join_url,
    open_portal,
    stop_pid,
    API_PID_PATH,
    EVAL_PID_PATH,
)

SMOKE_CASES = ROOT / "eval" / "data" / "sample" / "cases.jsonl"
MENU = """
  1  ask full          Full MedSwin (MAC + gate)
  2  ask naive         Naive-RAG control
  3  ask both          Side-by-side compare
  4  open clinician    Clinician CDS UI
  5  open dashboard    Ops / corpus dashboard
  6  open eval         Benchmark UI (starts it if needed)
  7  eval run          Smoke or TREC compare on the API
  8  index             Refresh embeddings and rebuild ANN
  9  status            Refresh health / corpus
  h  help              This menu
  q  quit              Leave servers running
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.operator",
        description="Operate the local MedSwin stack from one terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Commands
              console   Interactive operator (default)
              up        Start API (+ optional eval) and print / open portals
              ask       One question or the prompt REPL
              eval      Start the eval portal, or run a benchmark
              open      Open a web portal in the browser
              status    API, naive ready, corpus, eval
              index     Refresh embeddings and rebuild the ANN index
              stop      Stop API / eval processes this operator started

            Typical session
              ./scripts/start-local.sh
              → console starts the API, prints portal URLs, waits for a command
              → type a clinical question, or `open clinician`, or `eval run`
            """
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="console",
        choices=("console", "up", "ask", "eval", "open", "status", "index", "stop"),
    )
    parser.add_argument("--base-url", default=DEFAULT_API)
    parser.add_argument("--eval-url", default=DEFAULT_EVAL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--eval-port", type=int, default=8200)
    parser.add_argument("--mode", choices=MODES, default="both")
    parser.add_argument("--question", default="")
    parser.add_argument("--org-id", default="demo-org")
    parser.add_argument("--user-id", default="clinician-1")
    parser.add_argument("--patient-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--open", dest="open_browser", action="store_true")
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    parser.add_argument("--with-eval", action="store_true", help="Also start the eval portal")
    parser.add_argument("--portal", default="clinician", help="Portal name for `open`")
    parser.add_argument("--pipeline", choices=("medswin", "naive_rag", "both"), default="both")
    parser.add_argument("--cases-path", default=str(SMOKE_CASES))
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--eval-action", choices=("ui", "run"), default="ui")
    parser.set_defaults(open_browser=None)
    return parser


def _host_port(base_url: str, fallback_port: int) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if host in {"127.0.0.1", "localhost"}:
        bind_host = "0.0.0.0"
    else:
        bind_host = host
    return bind_host, parsed.port or fallback_port


def _prompt_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=args.base_url,
        mode=args.mode,
        question=args.question,
        org_id=args.org_id,
        user_id=args.user_id,
        patient_id=args.patient_id,
        session_id=args.session_id,
        top_k=args.top_k,
        timeout=args.timeout,
        json=args.json,
    )


def _print(text: str) -> None:
    print(text)


def cmd_status(args: argparse.Namespace) -> int:
    snapshot = collect_status(args.base_url, args.eval_url, args.org_id)
    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
        return 0
    print(format_status(snapshot))
    print()
    print(format_portals(args.base_url, args.eval_url))
    return 0 if (snapshot.get("api") or {}).get("ok") else 1


def cmd_up(args: argparse.Namespace) -> int:
    api_host, api_port = _host_port(args.base_url, args.port)
    api_state = ensure_api(args.base_url, host=args.host or api_host, port=args.port or api_port)
    print(f"API {api_state}: {args.base_url}")
    if args.with_eval:
        eval_host, eval_port = _host_port(args.eval_url, args.eval_port)
        eval_state = ensure_eval(args.eval_url, host=eval_host, port=eval_port)
        print(f"Eval {eval_state}: {args.eval_url}")
    print()
    print(format_portals(args.base_url, args.eval_url))
    print()
    cmd_status(args)
    should_open = True if args.open_browser is None else args.open_browser
    if args.command == "up" and args.open_browser is None:
        should_open = True
    if should_open:
        opened = ["clinician", "dashboard"]
        if args.with_eval:
            opened.append("eval")
        for name in opened:
            print(f"Opening {open_portal(name, args.base_url, args.eval_url)}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    ensure_api(args.base_url, host=args.host, port=args.port)
    prompt_args = _prompt_args(args)
    with httpx.Client(timeout=args.timeout) as client:
        if args.question:
            return run_once(client, prompt_args, args.question)
        from app.cli.prompt import repl

        return repl(client, prompt_args)


def cmd_open(args: argparse.Namespace) -> int:
    if args.portal in {"eval", "all"}:
        ensure_eval(args.eval_url, host=_host_port(args.eval_url, args.eval_port)[0], port=args.eval_port)
    elif args.portal != "eval":
        ensure_api(args.base_url, host=args.host, port=args.port)
    print(open_portal(args.portal, args.base_url, args.eval_url))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    ensure_api(args.base_url, host=args.host, port=args.port)
    print(f"Refreshing embeddings for org={args.org_id} ...")
    try:
        status, refresh = http_json(
            "POST",
            join_url(args.base_url, "/api/v1/storage/embeddings/refresh"),
            timeout=args.timeout,
            payload={"org_id": args.org_id},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Embedding refresh failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(refresh, indent=2, default=str) if args.json else f"refresh HTTP {status}")
    print(f"Building ANN index for org={args.org_id} ...")
    try:
        status, built = http_json(
            "POST",
            join_url(args.base_url, "/api/v1/storage/index/build"),
            timeout=args.timeout,
            payload={"force_rebuild": True, "org_id": args.org_id},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(built, indent=2, default=str))
    else:
        print(f"index HTTP {status}: {built.get('message') or built}")
    return 0 if status < 400 else 1


def cmd_eval(args: argparse.Namespace) -> int:
    ensure_api(args.base_url, host=args.host, port=args.port)
    if args.eval_action == "ui":
        eval_host, eval_port = _host_port(args.eval_url, args.eval_port)
        state = ensure_eval(args.eval_url, host=eval_host, port=eval_port)
        print(f"Eval portal {state}: {args.eval_url}")
        if args.open_browser is not False:
            print(open_portal("eval", args.base_url, args.eval_url))
        return 0

    from eval.app.config import Settings as EvalSettings
    from eval.app.runner import run_benchmark_sync
    from eval.app.schemas import RunRequest

    cases_path = args.cases_path
    if not cases_path:
        cases_path = str(SMOKE_CASES)
    print(f"Running eval pipeline={args.pipeline} cases={cases_path} max_cases={args.max_cases}")
    settings = EvalSettings(
        medswin_base_url=args.base_url,
        benchmark_org_id=os_benchmark_org(),
        benchmark_user_id=args.user_id,
    )
    request = RunRequest(
        cases_path=cases_path,
        max_cases=args.max_cases,
        pipeline=args.pipeline,
        top_k=args.top_k or 5,
        ingest_case_context=True,
    )
    try:
        run = run_benchmark_sync(request, settings)
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(run.model_dump(), indent=2, default=str))
        return 0
    print(f"run_id={run.run_id}")
    print(f"pipeline={run.config.get('pipeline')}")
    print(f"aggregate={run.aggregate}")
    comparison = (run.diagnostics or {}).get("pipeline_comparison")
    if comparison:
        print(f"naive_run_id={comparison.get('naive_run_id')}")
        print(f"delta={comparison.get('delta_medswin_minus_naive')}")
    print(f"Open the eval portal to inspect the audit: {args.eval_url}")
    return 0


def os_benchmark_org() -> str:
    import os

    return os.environ.get("BENCHMARK_ORG_ID", "bench-org")


def cmd_stop(args: argparse.Namespace) -> int:
    stopped = []
    if stop_pid(EVAL_PID_PATH):
        stopped.append("eval")
    if stop_pid(API_PID_PATH):
        stopped.append("api")
    if stopped:
        print("Stopped: " + ", ".join(stopped))
    else:
        print("No operator-started processes were running.")
    return 0


def _read(prompt: str, default: str = "") -> str:
    raw = input(prompt).strip()
    return raw or default


def _looks_like_question(raw: str) -> bool:
    if raw.lower() in {"exit", "quit", "q", "h", "help", "status", "menu"}:
        return False
    if raw[:1].isdigit() and len(raw) <= 2:
        return False
    return len(raw.split()) >= 3 or raw.endswith("?")


def cmd_console(args: argparse.Namespace) -> int:
    try:
        cmd_up(argparse.Namespace(**{**vars(args), "command": "console", "open_browser": bool(args.open_browser)}))
    except Exception as exc:  # noqa: BLE001
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    print()
    print("MedSwin operator. Type a clinical question, a menu number, or a command.")
    print(MENU)
    while True:
        try:
            raw = _read(f"operator [{args.mode}] > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        key = raw.lower()
        if key in {"q", "quit", "exit"}:
            print("Leaving servers running. Use `./scripts/start-local.sh stop` to shut them down.")
            return 0
        if key in {"h", "help", "menu", "?"}:
            print(MENU)
            continue
        if key in {"9", "status"}:
            cmd_status(args)
            continue
        if key in {"1", "ask full", "full"}:
            args.mode = "full"
            return_code = _ask_from_console(args)
            if return_code:
                print("Ask failed; console stays open.")
            continue
        if key in {"2", "ask naive", "naive"}:
            args.mode = "naive"
            return_code = _ask_from_console(args)
            if return_code:
                print("Ask failed; console stays open.")
            continue
        if key in {"3", "ask both", "both", "compare"}:
            args.mode = "both"
            return_code = _ask_from_console(args)
            if return_code:
                print("Ask failed; console stays open.")
            continue
        if key in {"4", "open clinician", "ui"}:
            args.portal = "clinician"
            cmd_open(args)
            continue
        if key in {"5", "open dashboard", "dashboard"}:
            args.portal = "dashboard"
            cmd_open(args)
            continue
        if key in {"6", "open eval", "eval ui"}:
            args.portal = "eval"
            args.eval_action = "ui"
            cmd_eval(args)
            continue
        if key in {"7", "eval", "eval run"}:
            args.eval_action = "run"
            try:
                cmd_eval(args)
            except Exception as exc:  # noqa: BLE001
                print(f"Eval failed: {exc}")
            continue
        if key in {"8", "index", "rebuild"}:
            cmd_index(args)
            continue
        if key.startswith("open "):
            args.portal = key.split(None, 1)[1]
            try:
                cmd_open(args)
            except Exception as exc:  # noqa: BLE001
                print(exc)
            continue
        if key.startswith("mode "):
            mode = key.split(None, 1)[1]
            if mode in MODES:
                args.mode = mode
                print(f"Default ask mode is now {mode}")
            else:
                print("Mode must be full, naive, or both.")
            continue
        if _looks_like_question(raw):
            args.question = raw
            code = cmd_ask(args)
            args.question = ""
            if code:
                print("Ask failed; console stays open.")
            continue
        print("Unknown command. Type `help` or a clinical question.")
    return 0


def _ask_from_console(args: argparse.Namespace) -> int:
    question = _read("Question: ")
    if not question:
        print("A question is required.")
        return 1
    if question.lower() in {"exit", "quit"}:
        return 0
    patient = _read(f"patient_id (optional, {args.patient_id or '-'}): ", args.patient_id)
    args.question = question
    args.patient_id = patient
    code = cmd_ask(args)
    args.question = ""
    return code


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.host == "0.0.0.0" and args.port == 8100:
        bind_host, bind_port = _host_port(args.base_url, 8100)
        args.host = bind_host
        args.port = bind_port
    dispatch = {
        "console": cmd_console,
        "up": cmd_up,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "open": cmd_open,
        "status": cmd_status,
        "index": cmd_index,
        "stop": cmd_stop,
    }
    try:
        return dispatch[args.command](args)
    except KeyboardInterrupt:
        print()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
