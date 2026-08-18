"""Local operator surfaces: portals, health probes, and process helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
API_PID_PATH = LOG_DIR / "medswin-api.pid"
EVAL_PID_PATH = LOG_DIR / "medswin-eval.pid"

DEFAULT_API = "http://127.0.0.1:8100"
DEFAULT_EVAL = "http://127.0.0.1:8200"

PORTAL_PATHS = {
    "clinician": ("/app/", "Clinician CDS — full MedSwin, naive-RAG, or both"),
    "dashboard": ("/api/v1/dashboard/", "Ops dashboard — corpus, ingest, models"),
    "docs": ("/docs", "OpenAPI explorer"),
    "health": ("/health", "API health"),
    "naive_ready": ("/api/v1/naive/ready", "Naive-RAG service preflight"),
}


def join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def portal_urls(api_base: str, eval_base: str) -> dict[str, str]:
    urls = {name: join_url(api_base, path) for name, (path, _hint) in PORTAL_PATHS.items()}
    urls["eval"] = eval_base.rstrip("/") + "/"
    return urls


def portal_catalog(api_base: str, eval_base: str) -> list[tuple[str, str, str]]:
    urls = portal_urls(api_base, eval_base)
    rows = [
        (name, urls[name], hint)
        for name, (_path, hint) in PORTAL_PATHS.items()
    ]
    rows.append(("eval", urls["eval"], "TREC / smoke benchmark UI (starts separately)"))
    return rows


def http_json(
    method: str,
    url: str,
    *,
    timeout: float = 4.0,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method.upper())
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        body: Any = {}
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = {"text": raw.decode("utf-8", errors="replace")}
        return int(response.status), body


def probe(url: str, *, timeout: float = 2.0, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        status, body = http_json("GET", url, timeout=timeout, params=params)
        return {"ok": 200 <= status < 400, "status": status, "body": body}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "error": str(exc)}


def collect_status(api_base: str, eval_base: str, org_id: str) -> dict[str, Any]:
    api = probe(join_url(api_base, "/health"))
    naive = probe(join_url(api_base, "/api/v1/naive/ready")) if api.get("ok") else {"ok": False, "error": "api_down"}
    storage = (
        probe(join_url(api_base, "/api/v1/storage/stats"), params={"org_id": org_id})
        if api.get("ok")
        else {"ok": False, "error": "api_down"}
    )
    eval_health = probe(join_url(eval_base, "/health"))
    return {
        "api_base": api_base,
        "eval_base": eval_base,
        "org_id": org_id,
        "api": api,
        "naive_ready": naive,
        "storage": storage,
        "eval": eval_health,
        "portals": portal_urls(api_base, eval_base),
    }


def format_status(snapshot: dict[str, Any]) -> str:
    api = snapshot.get("api") or {}
    naive = snapshot.get("naive_ready") or {}
    storage = snapshot.get("storage") or {}
    eval_health = snapshot.get("eval") or {}
    api_body = api.get("body") if isinstance(api.get("body"), dict) else {}
    naive_body = naive.get("body") if isinstance(naive.get("body"), dict) else {}
    stats = storage.get("body") if isinstance(storage.get("body"), dict) else {}
    sources = stats.get("source_counts") or {}
    lines = [
        f"API            : {'up' if api.get('ok') else 'down'}  {snapshot.get('api_base')}",
        f"  cloud_mode   : {api_body.get('cloud_mode', '-')}",
        f"  embeddings   : {api_body.get('embedding_model', '-')}",
        f"  reranker     : {api_body.get('reranker_model', '-')}",
        f"Naive ready    : {'up' if naive.get('ok') else 'down'}",
        f"  mongo        : {naive_body.get('mongo', '-')}",
        f"  chunks       : {naive_body.get('chunk_count', '-')}",
        f"  embedded     : {naive_body.get('embedded_count', '-')}",
        f"  index_file   : {naive_body.get('index_exists', '-')}",
        f"Storage ({snapshot.get('org_id')})",
        f"  chunks       : {stats.get('total_chunks', '-')}",
        f"  embeddings   : {stats.get('active_embeddings', stats.get('total_embeddings', '-'))}",
        f"  sources      : CPG={sources.get('CPG', 0)} EMR={sources.get('EMR', 0)} "
        f"LIT={sources.get('LIT', 0)} SAFETY={sources.get('SAFETY', 0)}",
        f"  index        : exists={stats.get('index_exists', '-')} "
        f"provenance={stats.get('index_provenance_valid', '-')}",
        f"Eval portal    : {'up' if eval_health.get('ok') else 'down'}  {snapshot.get('eval_base')}",
    ]
    if naive_body.get("chunk_count") and not naive_body.get("embedded_count"):
        lines.append("  warning      : chunks exist but 0 embeddings — refresh before asking or evaluating")
    if stats.get("index_exists") is False:
        lines.append("  warning      : ANN index is missing — naive will fall back to mongo_cosine")
    return "\n".join(lines)


def format_portals(api_base: str, eval_base: str) -> str:
    lines = ["Portals"]
    for name, url, hint in portal_catalog(api_base, eval_base):
        lines.append(f"  {name:12} {url}")
        lines.append(f"               {hint}")
    return "\n".join(lines)


def open_portal(name: str, api_base: str, eval_base: str) -> str:
    urls = portal_urls(api_base, eval_base)
    if name == "all":
        for key in ("clinician", "dashboard", "eval", "docs"):
            webbrowser.open(urls[key])
        return "opened clinician, dashboard, eval, docs"
    if name not in urls:
        known = ", ".join(sorted(urls))
        raise ValueError(f"Unknown portal {name!r}. Choose: {known}, all")
    webbrowser.open(urls[name])
    return urls[name]


def wait_healthy(url: str, *, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if probe(url, timeout=2.0).get("ok"):
            return True
        time.sleep(1)
    return False


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if _pid_running(pid) else None


def write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def stop_pid(path: Path) -> bool:
    pid = read_pid(path)
    if pid is None:
        if path.exists():
            path.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, 15)
    except OSError:
        path.unlink(missing_ok=True)
        return False
    for _ in range(20):
        if not _pid_running(pid):
            break
        time.sleep(0.15)
    if _pid_running(pid):
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    path.unlink(missing_ok=True)
    return True


def spawn_uvicorn(args: list[str], *, pid_path: Path, log_path: Path) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", *args],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_pid(pid_path, proc.pid)
    return proc.pid


def ensure_api(api_base: str, *, host: str, port: int) -> str:
    if probe(join_url(api_base, "/health")).get("ok"):
        return "reused"
    pid = spawn_uvicorn(
        ["app.main:app", "--host", host, "--port", str(port), "--reload"],
        pid_path=API_PID_PATH,
        log_path=LOG_DIR / "medswin-api.log",
    )
    if not wait_healthy(join_url(api_base, "/health"), timeout_s=60):
        raise RuntimeError(f"API at {api_base} did not become healthy (pid {pid}). See logs/medswin-api.log")
    return "started"


def ensure_eval(eval_base: str, *, host: str, port: int) -> str:
    if probe(join_url(eval_base, "/health")).get("ok"):
        return "reused"
    pid = spawn_uvicorn(
        ["eval.app.main:app", "--host", host, "--port", str(port)],
        pid_path=EVAL_PID_PATH,
        log_path=LOG_DIR / "medswin-eval.log",
    )
    if not wait_healthy(join_url(eval_base, "/health"), timeout_s=30):
        raise RuntimeError(f"Eval portal at {eval_base} did not become healthy (pid {pid}). See logs/medswin-eval.log")
    return "started"
