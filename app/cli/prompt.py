"""Interactive terminal client for the three MedSwin v2 pipelines.

Run after the API is healthy:

    python -m app.cli.prompt
    python -m app.cli.prompt --mode naive --question "Can metformin continue?"
    python -m app.cli.prompt --mode all --org-id demo-org
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Any, Optional

import httpx


MODES = ("naive", "two-stage", "full", "all")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt MedSwin and/or the naive-RAG baseline from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Modes
              naive      dense retrieval → generate
              two-stage  hybrid → rerank → gate → generate (no MAC)
              full       two-stage + MAC
              all        side-by-side comparison of all three

            Preferred local entry point:
              ./scripts/start-local.sh
            That opens the operator console (ask, portals). Benchmark results use system-eval.
            This module is the ask engine used by that console:
              ./scripts/start-local.sh ask --mode all --question "..."
              python -m app.cli.prompt --mode naive --question "..."
            """
        ),
    )
    parser.add_argument("--base-url", default="http://localhost:8100", help="MedSwin API origin")
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--question", default="", help="One-shot question. Omit for a REPL.")
    parser.add_argument("--org-id", default="demo-org")
    parser.add_argument("--user-id", default="clinician-1")
    parser.add_argument("--patient-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--top-k", type=int, default=None, help="Naive dense top-K override")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary")
    return parser


def _health(client: httpx.Client, base_url: str) -> dict[str, Any]:
    response = client.get(f"{base_url.rstrip('/')}/health")
    response.raise_for_status()
    return response.json()


def _payload(args: argparse.Namespace, question: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": question,
        "user_id": args.user_id,
        "org_id": args.org_id,
    }
    if args.patient_id:
        body["patient_id"] = args.patient_id
    if args.session_id:
        body["session_id"] = args.session_id
    if args.top_k is not None:
        body["constraints"] = {"top_k": args.top_k}
    return body


def _path(mode: str) -> str:
    if mode == "all":
        return "/api/v1/pipelines/compare"
    return f"/api/v1/pipelines/{mode}/chat"


def _wrap(label: str, text: str) -> str:
    body = textwrap.fill(text or "(empty)", width=96, subsequent_indent="  ")
    return f"{label}\n  {body}"


def _passage_lines(response: dict[str, Any]) -> str:
    bundle = response.get("evidence_bundle") or {}
    passages = bundle.get("passages") or []
    if not passages:
        return "  (no passages)"
    lines = []
    for item in passages[:12]:
        chunk_id = item.get("chunk_id")
        doc_id = item.get("doc_id")
        source = item.get("source_type")
        score = item.get("dense_score")
        if score is None:
            score = item.get("calibrated_score") or item.get("fusion_score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "-"
        lines.append(f"  - {chunk_id}  doc={doc_id}  source={source}  score={score_text}")
    if len(passages) > 12:
        lines.append(f"  … {len(passages) - 12} more")
    return "\n".join(lines)


def _print_single(title: str, response: dict[str, Any]) -> None:
    decision = response.get("policy_decision") or {}
    timing = response.get("timing_ms") or {}
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"pipeline          : {response.get('pipeline')}")
    print(f"trace_id          : {response.get('trace_id')}")
    print(f"retrieval_backend : {response.get('retrieval_backend')}")
    print(f"policy.passed     : {decision.get('passed')}")
    print(f"policy.action     : {decision.get('action')}")
    if timing:
        print(f"timing_ms         : {timing}")
    print()
    print(_wrap("ANSWER", str(response.get("answer") or "")))
    print()
    print("RETRIEVED PASSAGES")
    print(_passage_lines(response))
    notes = response.get("safety_notes")
    if notes:
        print()
        print(_wrap("NOTES", str(notes)))


def _print_compare(payload: dict[str, Any]) -> None:
    for pipeline in ("naive", "two-stage", "full"):
        _print_single(pipeline.upper(), (payload.get("results") or {}).get(pipeline) or {})
    diff = payload.get("adjacent_diffs") or {}
    print()
    print("=" * 88)
    print("DIFF")
    print("=" * 88)
    for key in (
        "naive_to_two_stage",
        "two_stage_to_full",
    ):
        print(f"{key:24}: {diff.get(key)}")


def run_once(client: httpx.Client, args: argparse.Namespace, question: str) -> int:
    url = f"{args.base_url.rstrip('/')}{_path(args.mode)}"
    try:
        response = client.post(url, json=_payload(args, question))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        if getattr(exc, "response", None) is not None:
            print(exc.response.text, file=sys.stderr)
        return 1
    data = response.json()
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    if args.mode == "all":
        _print_compare(data)
        if any((value.get("degraded_mode") or {}).get("error") for value in (data.get("results") or {}).values()):
            return 1
    else:
        title = args.mode.upper()
        _print_single(title, data)
        degraded = data.get("degraded_mode") or {}
        if degraded.get("error") or degraded.get("no_embeddings"):
            return 1
    print()
    return 0


def _read_line(prompt: str, default: str = "") -> str:
    raw = input(prompt).strip()
    return raw or default


def repl(client: httpx.Client, args: argparse.Namespace) -> int:
    print("Interactive MedSwin v2 prompt. Type 'exit' to quit.")
    print(f"API {args.base_url}  org={args.org_id}  default_mode={args.mode}")
    while True:
        try:
            mode = _read_line(f"Pipeline [{'/'.join(MODES)}] ({args.mode}): ", args.mode)
            if mode.lower() in {"exit", "quit"}:
                return 0
            if mode not in MODES:
                print(f"Unknown mode {mode!r}. Use {', '.join(MODES)}.")
                continue
            question = _read_line("Question: ")
            if not question:
                print("A question is required.")
                continue
            if question.lower() in {"exit", "quit"}:
                return 0
            patient = _read_line(f"patient_id (optional, current {args.patient_id or '-'}): ", args.patient_id)
            args.mode = mode
            args.patient_id = patient
            code = run_once(client, args, question)
            if code != 0:
                print("The request failed. The REPL stays open so you can retry.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    with httpx.Client(timeout=args.timeout) as client:
        try:
            health = _health(client, args.base_url)
        except Exception as exc:  # noqa: BLE001
            print(
                f"API health check failed at {args.base_url}/health: {exc}\n"
                "Start the runtime first: ./scripts/start-local.sh",
                file=sys.stderr,
            )
            return 2
        print(f"API healthy: {health}")
        if health.get("cloud_mode") is False and health.get("embedding_model") == "not_loaded":
            print(
                "Warning: local embedding model is not loaded. Naive-RAG will try "
                "EMBEDDING_URL and fail if that service is also down.",
                file=sys.stderr,
            )
        try:
            ready = client.get(f"{args.base_url.rstrip('/')}/api/v1/pipelines/ready")
            if ready.is_success:
                payload = ready.json()
                print(
                    "Naive ready: "
                    f"mongo={payload.get('mongo')} chunks={payload.get('chunk_count')} "
                    f"embedded={payload.get('embedded_count')} index={payload.get('index_exists')}"
                )
                if payload.get("chunk_count") and not payload.get("embedded_count"):
                    print(
                        "Warning: chunks exist but none have embeddings. "
                        "Ingest will try to attach vectors; otherwise POST "
                        "/api/v1/storage/embeddings/refresh then rebuild the index.",
                        file=sys.stderr,
                    )
        except Exception:
            pass
        if args.question:
            return run_once(client, args, args.question)
        return repl(client, args)


if __name__ == "__main__":
    raise SystemExit(main())
