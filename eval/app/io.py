from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import BenchmarkCase


def read_jsonl_cases(path: str | Path) -> list[BenchmarkCase]:
    path = Path(path)
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(BenchmarkCase.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid case JSONL at {path}:{line_no}: {exc}") from exc
    return cases


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
