"""Self-contained Markdown, HTML, JSON, and CSV benchmark reports."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

from .artifacts import atomic_json


def _rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = _rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(run_dir: Path, report: dict[str, Any], *, smoke: bool = False) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    primary = report.get("primary_endpoints", [])
    operational = report.get("operational", [])
    write_csv(tables / "primary_endpoints.csv", primary)
    write_csv(tables / "operational.csv", operational)
    write_csv(tables / "retrieval.csv", report.get("retrieval", []))
    write_csv(tables / "answers.csv", report.get("answers", []))
    write_csv(tables / "stage_effects.csv", report.get("stage_effects", []))
    atomic_json(run_dir / "report.json", report)
    banner = "SMOKE TEST — NOT BENCHMARK RESULTS\n\n" if smoke else ""
    markdown = [f"# MedSwin system evaluation\n\n{banner}"]
    markdown.append("## Primary endpoints\n")
    markdown.append(_markdown_table(primary))
    markdown.append("\n## Adjacent-stage effects\n")
    markdown.append(_markdown_table(report.get("stage_effects", [])))
    markdown.append("\n## Limitations\n")
    markdown.append(
        "LIT-only retrieval, incomplete TREC pooled judgments, a derived closed BioASQ corpus, automated NLI diagnostics, possible historic-data contamination, and no clinician validation.\n"
    )
    markdown.append(f"\nManifest: `manifest.json`\n")
    (run_dir / "report.md").write_text("".join(markdown), encoding="utf-8")
    escaped = html.escape((run_dir / "report.md").read_text(encoding="utf-8"))
    watermark = '<div class="watermark">SMOKE TEST — NOT BENCHMARK RESULTS</div>' if smoke else ""
    (run_dir / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>MedSwin system evaluation</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:1100px}.watermark{color:#a00;font-weight:800;font-size:1.4rem}</style>"
        f"{watermark}<pre>{escaped}</pre>",
        encoding="utf-8",
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No clean completed observations.\n"
    columns = sorted({key for row in rows for key in row})
    header = "| " + " | ".join(columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |\n" for row in rows)
    return header + divider + body
