"""Official NIST CDS 2016 files. Never score ir_datasets qrels as inferred measures."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .contract import EXPECTED_QRELS, EXPECTED_SAMPLEVAL_LINES, PACKAGE_ROOT

NIST_DIR = Path(os.environ.get("NIST_DIR") or PACKAGE_ROOT / "nist")
BIN_DIR = PACKAGE_ROOT / "bin"

NIST_FILES = {
    "topics2016.xml": {
        "url": "https://trec.nist.gov/data/clinical/topics2016.xml",
        "sha256": "167541d16ab0986fd36045fb4e1104fccb6c8df18e1842b7ee448e7639479767",
    },
    "qrels-treceval-2016.txt": {
        "url": "https://trec.nist.gov/data/clinical/qrels-treceval-2016.txt",
        "sha256": "285fcf088b81ea3ad926b054d92aaddf24b148287f9f574c8edccbf21b5ed3ac",
        "lines": EXPECTED_QRELS,
    },
    "qrels-sampleval-2016.txt": {
        "url": "https://trec.nist.gov/data/clinical/qrels-sampleval-2016.txt",
        "sha256": "f3617dcdd37b00aae48a943e437a2059fe629c11fb737fc39530a29f41dd82e2",
        "lines": EXPECTED_SAMPLEVAL_LINES,
    },
    "sample_eval.pl": {
        "url": "https://trec.nist.gov/data/clinical/sample_eval.pl",
        "sha256": "8af44fab50fed7ae8d00c75cc0358ccd7242274f213d9dfa805fa64c74c069f4",
    },
}

TREC_EVAL_RELEASE = "https://github.com/usnistgov/trec_eval/archive/refs/tags/v9.0.8.tar.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_nist(directory: Path | None = None) -> dict[str, str]:
    root = directory or NIST_DIR
    verified: dict[str, str] = {}
    for name, spec in NIST_FILES.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Missing NIST file: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {name}: {digest} != {spec['sha256']}")
        expected_lines = spec.get("lines")
        if expected_lines is not None:
            lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
            if lines != expected_lines:
                raise RuntimeError(f"{name} has {lines} lines; expected {expected_lines}")
        verified[name] = digest
    return verified


def download_nist(directory: Path | None = None, force: bool = False) -> dict[str, str]:
    root = directory or NIST_DIR
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in NIST_FILES.items():
        path = root / name
        if path.exists() and not force:
            continue
        tmp = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.replace(path)
    return verify_nist(root)


def nist_paths(directory: Path | None = None) -> dict[str, Path]:
    root = directory or NIST_DIR
    verify_nist(root)
    return {name: root / name for name in NIST_FILES}


def find_trec_eval() -> Path:
    explicit = (os.environ.get("TREC_EVAL_BIN") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise RuntimeError(f"TREC_EVAL_BIN is not a file: {path}")
        return path
    on_path = shutil.which("trec_eval")
    if on_path:
        return Path(on_path)
    local = BIN_DIR / "trec_eval"
    if local.is_file():
        return local
    raise RuntimeError(
        "trec_eval not found. Set TREC_EVAL_BIN or run benchmarks.trec_cds2016.nist.ensure_trec_eval()"
    )


def ensure_trec_eval() -> Path:
    try:
        return find_trec_eval()
    except RuntimeError:
        pass
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    extracted = [path for path in BIN_DIR.glob("trec_eval-*") if path.is_dir()]
    if not extracted:
        archive = BIN_DIR / "trec_eval-v9.0.8.tar.gz"
        if not archive.is_file():
            urllib.request.urlretrieve(TREC_EVAL_RELEASE, archive)
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(BIN_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        extracted = [path for path in BIN_DIR.glob("trec_eval-*") if path.is_dir()]
    if not extracted:
        raise RuntimeError(f"trec_eval source directory missing after extract in {BIN_DIR}")
    src = extracted[0]
    built = src / "trec_eval"
    if not built.is_file():
        subprocess.run(["make"], cwd=src, check=True, capture_output=True, text=True)
    dest = BIN_DIR / "trec_eval"
    shutil.copy2(built, dest)
    dest.chmod(0o755)
    return dest
