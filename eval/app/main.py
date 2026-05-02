from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .io import read_jsonl_cases
from .runner import run_benchmark
from .schemas import RunRequest

app = FastAPI(title="MedSwin End-to-End System Benchmark", version="0.1.0")
settings = get_settings()

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = static_dir / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "medswin-system-benchmark"}


@app.get("/api/cases")
async def list_cases(path: str = "data/sample/cases.jsonl") -> dict:
    try:
        cases = read_jsonl_cases(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "path": path,
        "count": len(cases),
        "preview": [c.model_dump(exclude={"patient_context"}) for c in cases[:5]],
    }


@app.post("/api/run")
async def run(req: RunRequest) -> JSONResponse:
    try:
        audit = await run_benchmark(req, settings)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(audit.model_dump())


@app.get("/api/runs")
async def list_runs() -> dict:
    run_dir = Path(settings.run_store_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append({
                "run_id": data.get("run_id", path.stem),
                "created_at": data.get("created_at"),
                "dataset": data.get("dataset"),
                "num_cases": data.get("num_cases"),
                "mean_msas": (data.get("aggregate") or {}).get("mean_msas"),
            })
        except Exception:  # noqa: BLE001
            continue
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    path = Path(settings.run_store_dir) / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/download")
async def download_run(run_id: str) -> FileResponse:
    path = Path(settings.run_store_dir) / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return FileResponse(str(path), filename=f"medswin-system-audit-{run_id}.json")
