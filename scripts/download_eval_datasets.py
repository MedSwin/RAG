#!/usr/bin/env python3
"""Download and materialize official TREC-CDS 2016 and BioASQ Task B local datasets.

Outputs (gitignored under ./data/benchmarks/):
1. data/benchmarks/trec_cds2016_judged/
   - Official NIST 2016 clinical topics (30 cases: diagnosis, test, treatment)
   - Official NIST qrelsverified via SHA-256 + compiled trec_eval v9.0.8
   - Real PubMed Central clinical articles judged in NIST qrels (grades 2, 1, and 0
     hard negatives across all 30 topics) fetched from Europe PMC REST API, plus
     structured EMR admission notes for patients trec-cds-1..trec-cds-30.
2. data/benchmarks/bioasq_real/
   - Stratified BioASQ Task B questions across all 4 official types
     (yesno, factoid, list, summary) with gold PMIDs, snippets, exact_answer,
     and ideal_answer from BioASQ training11b + PubMed passage corpus from
     rag-datasets/rag-mini-bioasq.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.trec_cds2016.nist import NIST_DIR, download_nist, ensure_trec_eval
from benchmarks.trec_cds2016.topics import load_topics


def _sha256_obj(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _classify_pmc_source_type(title: str, abstract: str, pub_types: list[str]) -> str:
    joined = f"{title} {abstract} {' '.join(pub_types)}".lower()
    if any(k in joined for k in ("guideline", "practice parameter", "consensus statement", "clinical recommendation")):
        return "CPG"
    if any(k in joined for k in ("contraindicat", "adverse drug", "black box", "hepatotox", "nephrotox", "anaphylax", "toxicity")):
        return "SAFETY"
    return "LIT"


def materialize_trec_cds2016(
    out_dir: Path,
    *,
    per_topic_grade2: int = 5,
    per_topic_grade1: int = 4,
    per_topic_grade0: int = 4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and (out_dir / "corpus.json").exists() and (out_dir / "topics.json").exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    nist_hashes = download_nist(NIST_DIR)
    trec_eval_bin = ensure_trec_eval()
    topics = load_topics(NIST_DIR / "topics2016.xml")

    by_topic_grade: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    full_qrels: dict[str, dict[str, int]] = defaultdict(dict)
    qrels_path = NIST_DIR / "qrels-treceval-2016.txt"
    with qrels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            topic_num = int(parts[0])
            doc_id = parts[2].removeprefix("PMC")
            rel = int(parts[3])
            by_topic_grade[topic_num][rel].append(doc_id)
            full_qrels[str(topic_num)][doc_id] = rel

    selected_pmcids: list[str] = []
    seen_pmcids: set[str] = set()
    for topic in topics:
        buckets = by_topic_grade[topic.number]
        for grade, quota in ((2, per_topic_grade2), (1, per_topic_grade1), (0, per_topic_grade0)):
            candidates = sorted(
                buckets.get(grade, []),
                key=lambda d: hashlib.sha256(f"{topic.number}:{grade}:{d}".encode()).hexdigest(),
            )
            added = 0
            for doc_id in candidates:
                if doc_id not in seen_pmcids:
                    seen_pmcids.add(doc_id)
                    selected_pmcids.append(doc_id)
                added += 1
                if added >= quota:
                    break

    fetched: dict[str, dict[str, Any]] = {}
    headers = {"User-Agent": "MedSwin-CDSS-Benchmark/2.0"}
    with httpx.Client(timeout=30.0, headers=headers) as client:
        batch_size = 20
        for start in range(0, len(selected_pmcids), batch_size):
            batch = selected_pmcids[start : start + batch_size]
            query = " OR ".join(f"PMCID:PMC{doc_id}" for doc_id in batch)
            resp = client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": query, "resultType": "core", "pageSize": 50, "format": "json"},
            )
            resp.raise_for_status()
            rows = resp.json().get("resultList", {}).get("result", [])
            for item in rows:
                pmcid_raw = str(item.get("pmcid") or "").strip()
                if not pmcid_raw.startswith("PMC"):
                    continue
                doc_id = pmcid_raw.removeprefix("PMC")
                title = re.sub(r"<[^>]+>", "", str(item.get("title") or "")).strip()
                abstract = re.sub(r"<[^>]+>", "", str(item.get("abstractText") or "")).strip()
                if not (title or abstract):
                    continue
                pub_types_raw = (item.get("pubTypeList") or {}).get("pubType") or []
                if isinstance(pub_types_raw, str):
                    pub_types = [pub_types_raw]
                else:
                    pub_types = [str(p) for p in pub_types_raw if p]
                mesh_headings = [
                    str(m.get("descriptorName"))
                    for m in ((item.get("meshHeadingList") or {}).get("meshHeading") or [])
                    if isinstance(m, dict) and m.get("descriptorName")
                ]
                fetched[doc_id] = {
                    "doc_id": doc_id,
                    "pmcid": pmcid_raw,
                    "pmid": str(item.get("pmid") or ""),
                    "title": title,
                    "abstract": abstract,
                    "journal": str((item.get("journalInfo") or {}).get("journal", {}).get("title") or ""),
                    "pub_year": str(item.get("pubYear") or ""),
                    "pub_types": pub_types,
                    "mesh_terms": mesh_headings[:12],
                    "source_type": _classify_pmc_source_type(title, abstract, pub_types),
                }
            time.sleep(0.08)

    corpus_docs: list[dict[str, Any]] = []
    for doc_id in sorted(fetched):
        rec = fetched[doc_id]
        mesh_line = f"MeSH terms: {', '.join(rec['mesh_terms'])}." if rec["mesh_terms"] else ""
        text = "\n\n".join(part for part in (rec["title"], rec["abstract"], mesh_line) if part)
        corpus_docs.append(
            {
                "doc_id": doc_id,
                "pmcid": rec["pmcid"],
                "pmid": rec["pmid"],
                "title": rec["title"],
                "source_type": rec["source_type"],
                "text": text,
                "metadata": {
                    "dataset": "trec-cds-2016-judged",
                    "pmcid": rec["pmcid"],
                    "pmid": rec["pmid"],
                    "journal": rec["journal"],
                    "pub_year": rec["pub_year"],
                    "pub_types": rec["pub_types"],
                    "mesh_terms": rec["mesh_terms"],
                },
            }
        )

    emr_docs: list[dict[str, Any]] = []
    topic_records: list[dict[str, Any]] = []
    qrels_subset_lines: list[str] = []
    for topic in topics:
        t_key = str(topic.number)
        topic_gains: dict[str, int] = {}
        for doc_id in sorted(fetched):
            if doc_id in full_qrels[t_key]:
                rel = full_qrels[t_key][doc_id]
                qrels_subset_lines.append(f"{topic.number} 0 {doc_id} {rel}")
                if rel > 0:
                    topic_gains[doc_id] = rel
        topic_records.append(
            {
                "number": topic.number,
                "topic_type": topic.topic_type,
                "patient_id": topic.patient_id,
                "type_question": topic.type_question,
                "note": topic.note,
                "description": topic.description,
                "summary": topic.summary,
                "query_note": topic.t1_query("note"),
                "query_summary": topic.t1_query("summary"),
                "relevant_doc_gains": topic_gains,
            }
        )
        emr_docs.append(
            {
                "doc_id": f"emr-{topic.patient_id}",
                "patient_id": topic.patient_id,
                "title": f"Admission Clinical Note ({topic.patient_id}, {topic.topic_type})",
                "source_type": "EMR",
                "text": f"Patient ID: {topic.patient_id}\nClinical Summary: {topic.summary}\n\nAdmission Note:\n{topic.note}",
                "metadata": {
                    "dataset": "trec-cds-2016-emr",
                    "topic_number": topic.number,
                    "topic_type": topic.topic_type,
                    "benchmark_patient_context": True,
                },
            }
        )

    _atomic_json(out_dir / "topics.json", topic_records)
    _atomic_json(out_dir / "corpus.json", corpus_docs)
    _atomic_json(out_dir / "emr_notes.json", emr_docs)
    (out_dir / "qrels_subset.txt").write_text("\n".join(qrels_subset_lines) + "\n", encoding="utf-8")

    manifest = {
        "dataset": "trec-cds-2016-judged",
        "nist_verified_sha256": nist_hashes,
        "trec_eval_bin": str(trec_eval_bin),
        "topic_count": len(topic_records),
        "topics_with_relevant_fetched_docs": sum(1 for t in topic_records if t["relevant_doc_gains"]),
        "pmc_requested_count": len(selected_pmcids),
        "pmc_fetched_count": len(corpus_docs),
        "emr_doc_count": len(emr_docs),
        "source_type_counts": {
            st: sum(1 for d in corpus_docs if d["source_type"] == st) for st in ("LIT", "CPG", "SAFETY")
        },
        "corpus_sha256": _sha256_obj(corpus_docs),
        "topics_sha256": _sha256_obj(topic_records),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def materialize_bioasq(
    out_dir: Path,
    *,
    per_type: int = 25,
    distractor_passages: int = 250,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and (out_dir / "corpus.json").exists() and (out_dir / "questions.json").exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("question_count", 0) > 0:
            return cached

    token = os.getenv("HF_TOKEN") or None
    repo_id = "rag-datasets/rag-mini-bioasq"
    train_json_path = hf_hub_download(repo_id=repo_id, filename="raw_data/training11b.json", repo_type="dataset", token=token)
    passages_parquet_path = hf_hub_download(
        repo_id=repo_id, filename="data/passages.parquet/part.0.parquet", repo_type="dataset", token=token
    )

    passages_df = pd.read_parquet(passages_parquet_path)
    passage_by_id: dict[str, str] = {}
    for pid_raw, row in passages_df.iterrows():
        pid = str(pid_raw).strip()
        text = str(row.get("passage", "")).strip()
        if pid and text and text.lower() != "nan":
            passage_by_id[pid] = text

    raw_payload = json.loads(Path(train_json_path).read_text(encoding="utf-8"))
    raw_questions = raw_payload.get("questions") if isinstance(raw_payload, dict) else raw_payload

    valid_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_questions:
        qid = str(row.get("id") or "").strip()
        body = " ".join(str(row.get("body") or "").split())
        qtype = str(row.get("type") or "").strip().lower()
        if not (qid and body and qtype in {"yesno", "factoid", "list", "summary"}):
            continue
        if qtype != "summary" and row.get("exact_answer") is None:
            continue
        pmids = sorted(
            {
                str(doc).rsplit("/", 1)[-1].removeprefix("PMID-")
                for doc in (row.get("documents") or [])
                if str(doc).rsplit("/", 1)[-1].removeprefix("PMID-") in passage_by_id
            }
        )
        if not pmids:
            continue
        ideal = row.get("ideal_answer")
        if isinstance(ideal, list):
            ideal = " ".join(str(x) for x in ideal if x)
        ideal = str(ideal or "").strip()
        if not ideal:
            continue
        valid_by_type[qtype].append(
            {
                "dataset": "bioasq-11b-rag",
                "case_id": qid,
                "query": body,
                "question_type": qtype,
                "pmids": pmids,
                "gains": {pmid: 1 for pmid in pmids},
                "exact_answer": row.get("exact_answer"),
                "ideal_answer": ideal,
                "gold_snippets": [
                    " ".join(str(s.get("text") or "").split())
                    for s in (row.get("snippets") or [])
                    if isinstance(s, dict) and s.get("text")
                ],
            }
        )

    sampled_questions: list[dict[str, Any]] = []
    gold_pmids: set[str] = set()
    for qtype in ("yesno", "factoid", "list", "summary"):
        stratum = sorted(
            valid_by_type[qtype],
            key=lambda item: hashlib.sha256(item["case_id"].encode("utf-8")).hexdigest(),
        )[:per_type]
        sampled_questions.extend(stratum)
        for item in stratum:
            gold_pmids.update(item["pmids"])

    distractor_candidates = sorted(
        (pid for pid in passage_by_id if pid not in gold_pmids),
        key=lambda pid: hashlib.sha256(f"bioasq-distractor:{pid}".encode("utf-8")).hexdigest(),
    )[:distractor_passages]

    chosen_pmids = sorted(gold_pmids | set(distractor_candidates))
    corpus_docs: list[dict[str, Any]] = []
    for pmid in chosen_pmids:
        text = passage_by_id[pmid]
        lowered = text.lower()
        if any(k in lowered for k in ("guideline", "consensus", "recommended first-line")):
            source_type = "CPG"
        elif any(k in lowered for k in ("contraindicat", "adverse effect", "toxicity", "hepatotox", "nephrotox")):
            source_type = "SAFETY"
        else:
            source_type = "LIT"
        corpus_docs.append(
            {
                "doc_id": pmid,
                "pmid": pmid,
                "title": f"PubMed Passage PMID:{pmid}",
                "source_type": source_type,
                "text": text,
                "metadata": {
                    "dataset": "bioasq-11b-rag",
                    "pmid": pmid,
                    "is_gold_for_sample": pmid in gold_pmids,
                },
            }
        )

    _atomic_json(out_dir / "questions.json", sampled_questions)
    _atomic_json(out_dir / "corpus.json", corpus_docs)
    manifest = {
        "dataset": "bioasq-11b-rag",
        "hf_source_repo": repo_id,
        "question_count": len(sampled_questions),
        "strata_counts": {
            qt: sum(1 for q in sampled_questions if q["question_type"] == qt)
            for qt in ("yesno", "factoid", "list", "summary")
        },
        "gold_pmid_count": len(gold_pmids),
        "distractor_pmid_count": len(distractor_candidates),
        "corpus_doc_count": len(corpus_docs),
        "source_type_counts": {
            st: sum(1 for d in corpus_docs if d["source_type"] == st) for st in ("LIT", "CPG", "SAFETY")
        },
        "questions_sha256": _sha256_obj(sampled_questions),
        "corpus_sha256": _sha256_obj(corpus_docs),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    base = REPO_ROOT / "data" / "benchmarks"
    trec_manifest = materialize_trec_cds2016(base / "trec_cds2016_judged")
    bioasq_manifest = materialize_bioasq(base / "bioasq_real")
    print(
        json.dumps(
            {
                "trec_cds2016_judged": trec_manifest,
                "bioasq_real": bioasq_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
