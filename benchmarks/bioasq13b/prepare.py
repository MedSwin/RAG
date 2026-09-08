"""Non-interactive BioASQ 13b acquisition, deterministic sampling, and NCBI cache."""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from benchmarks.system_eval.artifacts import atomic_json, digest, read_json
from benchmarks.system_eval.contracts import SAMPLING_ALGORITHM_VERSION

LOGIN_URL = "https://participants-area.bioasq.org/accounts/login/"
DATASETS_URL = "https://participants-area.bioasq.org/datasets/"
CSRF = re.compile(r'<input[^>]+name=["\']csrf(?:middleware)?token["\'][^>]+value=["\']([^"\']+)', re.I)
# Prefer Training 13b (≈5.3k questions). The "13b golden enriched" release is the
# small test set (~85 questions) and cannot support the 30-per-type sample.
LINK = re.compile(r'href=["\']([^"\']+)["\'][^>]*>([^<]*Training[^<]*13b[^<]*)<', re.I)


def _sha(data: bytes | str) -> str:
    return sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _normalise_question(row: dict[str, Any]) -> dict[str, Any] | None:
    qid = str(row.get("id") or row.get("question_id") or "").strip()
    body = " ".join(str(row.get("body") or row.get("question") or "").split())
    question_type = str(row.get("type") or row.get("question_type") or "").lower().strip()
    documents = [str(value).rsplit("/", 1)[-1] for value in row.get("documents", []) if str(value).rsplit("/", 1)[-1].startswith("PMID-") or str(value).rsplit("/", 1)[-1].isdigit()]
    pmids = [value.removeprefix("PMID-") for value in documents]
    ideal = row.get("ideal_answer") or row.get("ideal_answers")
    if isinstance(ideal, list):
        ideal = " ".join(str(value) for value in ideal)
    if not (qid and body and question_type in {"yesno", "factoid", "list", "summary"} and pmids and ideal):
        return None
    # Summary items may omit exact answers; other types require an exact field.
    if question_type != "summary" and row.get("exact_answer") is None:
        return None
    return {
        "id": qid,
        "body": body,
        "question_type": question_type,
        "pmids": sorted(set(pmids)),
        "exact_answer": row.get("exact_answer"),
        "ideal_answer": str(ideal),
        "snippets": row.get("snippets") or [],
    }


def _download_source(cache: Path, *, username: str, password: str) -> tuple[list[dict[str, Any]], str]:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        login = client.get(LOGIN_URL)
        login.raise_for_status()
        token = CSRF.search(login.text)
        if not token:
            raise RuntimeError("BioASQ login page did not contain a CSRF field")
        result = client.post(LOGIN_URL, data={"username": username, "password": password, "csrfmiddlewaretoken": token.group(1)}, headers={"Referer": LOGIN_URL})
        if "login" in str(result.url).lower() or not result.is_success:
            raise RuntimeError("BioASQ authentication failed")
        page = client.get(DATASETS_URL)
        page.raise_for_status()
        matches = LINK.findall(page.text)
        if len(matches) != 1:
            raise RuntimeError("Could not discover exactly one BioASQ Training 13b download")
        archive = client.get(urljoin(DATASETS_URL, matches[0][0]))
        archive.raise_for_status()
        body = archive.content
        if "login" in str(archive.url).lower() or body.lstrip().lower().startswith(b"<!doctype html"):
            raise RuntimeError("BioASQ download resolved to authentication HTML")
        payload = _parse_bioasq_payload(body)
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError("BioASQ payload does not contain questions")
    source_hash = _sha(body)
    cache.mkdir(parents=True, exist_ok=True)
    # Never persist the redistributable source archive. The derived immutable
    # manifest records only its hash and the selected-question metadata.
    atomic_json(cache / "source.manifest.json", {"edition": "13b", "kind": "training", "source_sha256": source_hash})
    return rows, source_hash


def _parse_bioasq_payload(body: bytes) -> Any:
    """Parse BioASQ training/golden payload from raw JSON or a ZIP archive."""
    stripped = body.lstrip()
    if stripped[:1] in (b"{", b"["):
        try:
            return json.loads(body)
        except ValueError as exc:
            raise RuntimeError("BioASQ download is not JSON") from exc
    if stripped.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            json_names = [name for name in archive.namelist() if name.lower().endswith(".json") and not name.endswith("/")]
            if not json_names:
                raise RuntimeError("BioASQ ZIP archive contains no JSON payload")
            # Prefer the largest JSON member (full set vs tiny sidecars).
            chosen = max(json_names, key=lambda name: archive.getinfo(name).file_size)
            try:
                return json.loads(archive.read(chosen))
            except ValueError as exc:
                raise RuntimeError(f"BioASQ ZIP member {chosen!r} is not JSON") from exc
    raise RuntimeError("BioASQ download is neither JSON nor ZIP")


def deterministic_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised = [_normalise_question(row) for row in rows]
    unique = {row["id"]: row for row in normalised if row}
    selected = []
    for question_type in ("yesno", "factoid", "list", "summary"):
        stratum = sorted((row for row in unique.values() if row["question_type"] == question_type), key=lambda row: _sha(row["id"]))
        if len(stratum) < 30:
            raise RuntimeError(f"BioASQ 13b has fewer than 30 valid {question_type} questions")
        selected.extend(stratum[:30])
    return selected


def _ncbi_snapshot(cache: Path, pmids: list[str]) -> dict[str, dict[str, str]]:
    destination = cache / "ncbi_snapshot.json"
    if destination.exists():
        return read_json(destination)
    email = os.getenv("NCBI_EMAIL")
    if not email:
        raise RuntimeError("NCBI_EMAIL is required to build the BioASQ closed corpus")
    values: dict[str, dict[str, str]] = {}
    with httpx.Client(timeout=60) as client:
        for offset in range(0, len(pmids), 100):
            batch = pmids[offset : offset + 100]
            response = client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "email": email, "api_key": os.getenv("NCBI_API_KEY", "")})
            response.raise_for_status()
            for pmid in batch:
                block = re.search(rf"<PubmedArticle>.*?<PMID[^>]*>{re.escape(pmid)}</PMID>.*?</PubmedArticle>", response.text, re.S)
                if not block:
                    continue
                title = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", block.group(0), re.S)
                abstract = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", block.group(0), re.S)
                values[pmid] = {"title": re.sub(r"<[^>]+>", "", title.group(1) if title else "").strip(), "abstract": " ".join(re.sub(r"<[^>]+>", "", part).strip() for part in abstract)}
            time.sleep(0.34)
    atomic_json(destination, values)
    atomic_json(cache / "ncbi_snapshot.manifest.json", {"sha256": digest(values), "count": len(values)})
    return values


def prepare_bioasq(*, cache_dir: Path, credentials_required: bool = True) -> list[dict[str, Any]]:
    manifest_path = cache_dir / "sample.manifest.json"
    sample_path = cache_dir / "sample.json"
    if sample_path.exists() and manifest_path.exists():
        return read_json(sample_path)
    username, password = os.getenv("BIOASQ_USERNAME"), os.getenv("BIOASQ_PASSWORD")
    if credentials_required and not (username and password):
        raise RuntimeError("BIOASQ_USERNAME and BIOASQ_PASSWORD are required; manual download is not supported")
    if not (username and password):
        raise RuntimeError("BioASQ sample is not prepared and credentials are required")
    rows, source_hash = _download_source(cache_dir, username=username, password=password)
    sampled = deterministic_sample(rows)
    snapshot = _ncbi_snapshot(cache_dir, sorted({pmid for row in sampled for pmid in row["pmids"]}))
    corpus = [
        {"doc_id": pmid, "source_type": "LIT", "text": "\n\n".join(part for part in (item.get("title"), item.get("abstract")) if part)}
        for pmid, item in sorted(snapshot.items())
        if item.get("title") or item.get("abstract")
    ]
    atomic_json(cache_dir / "closed_corpus.json", corpus)
    cases = [{
        "dataset": "bioasq",
        "case_id": row["id"],
        "query": row["body"],
        "gains": {pmid: 1 for pmid in row["pmids"]},
        "question_type": row["question_type"],
        "exact_answer": row["exact_answer"],
        "ideal_answer": row["ideal_answer"],
        "gold_snippets": [" ".join(str(snippet.get("text") or "").split()) for snippet in row["snippets"] if snippet.get("text")],
    } for row in sampled]
    atomic_json(sample_path, cases)
    atomic_json(manifest_path, {"source_sha256": source_hash, "sample_sha256": digest(cases), "sampling_algorithm": SAMPLING_ALGORITHM_VERSION, "ids": [row["case_id"] for row in cases], "ncbi_snapshot_sha256": digest(snapshot), "closed_corpus_sha256": digest(corpus), "metadata": {"count": 120, "strata": {kind: 30 for kind in ("yesno", "factoid", "list", "summary")}}})
    return cases
