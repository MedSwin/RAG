#!/usr/bin/env python3
"""Persist immutable T3 packs. Product path: note as EMR, type question as query."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from benchmarks.expert.claims import ABSTAIN_CLAIM_CAP, CLAIM_CAP, attach_citations, split_claims
from benchmarks.trec_cds2016.topics import load_topics

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKS_DIR = PACKAGE_ROOT / "packs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductClient:
    def __init__(self, base_url: str, org_id: str, user_id: str, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.user_id = user_id
        self.timeout_s = timeout_s

    async def ingest_note(self, client: httpx.AsyncClient, topic) -> None:
        excerpt = " ".join(topic.note.split()[:900])
        payload = [
            {
                "doc_id": f"trec-cds-2016:{topic.number}:note",
                "title": f"TREC CDS case note {topic.number}",
                "version": "t3",
                "patient_id": topic.patient_id,
                "source_reliability": 0.8,
                "text": topic.note,
                "chunks": [
                    {
                        "chunk_id": f"trec-cds-2016:{topic.number}:note_chunk_0",
                        "text": excerpt,
                        "section": "patient_context",
                    }
                ],
            }
        ]
        response = await client.post(
            f"{self.base_url}/api/v1/medswin/ingest",
            params={"source_type": "EMR", "org_id": self.org_id},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()

    async def chat(self, client: httpx.AsyncClient, topic, pipeline: str, condition: str = "full") -> dict:
        path = "/api/v1/naive/chat" if pipeline == "naive" else "/api/v1/medswin/chat"
        constraints = {
            "source_policy": "ANY",
            "clinical_scope": "clinician_cds",
            "disable_patient_retrieval_context": True,
            "include_patient_context_in_query": False,
            "disable_gate": condition == "no_gate",
            "disable_mac": condition == "no_mac",
        }
        payload = {
            "org_id": self.org_id,
            "user_id": self.user_id,
            "patient_id": topic.patient_id,
            "query": topic.t3_query(),
            "constraints": constraints,
        }
        response = await client.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()


def _passages(response: dict) -> list[dict]:
    bundle = response.get("evidence_bundle") or {}
    items = []
    for key in ("passages", "evidence", "selected_passages", "chunks", "items"):
        raw = bundle.get(key)
        if isinstance(raw, list):
            items.extend(item for item in raw if isinstance(item, dict))
    return items


def _answered(response: dict, pipeline: str) -> bool:
    if pipeline == "naive":
        return True
    policy = response.get("policy_decision") or {}
    if isinstance(policy, dict) and policy.get("passed") is False:
        return False
    action = str((response.get("sufficiency_decision") or {}).get("action") or "").lower()
    if action in {"insufficient_evidence", "abstain", "insufficient"}:
        return False
    return bool(str(response.get("answer") or "").strip())


def build_pack(topic, pipeline: str, generator: str, response: dict) -> dict:
    passages = _passages(response)
    answered = _answered(response, pipeline)
    answer = str(response.get("answer") or "")
    cap = CLAIM_CAP if answered else ABSTAIN_CLAIM_CAP
    claims = attach_citations(split_claims(answer, cap=cap), passages)
    return {
        "created_at": _now(),
        "topic_id": topic.number,
        "topic_type": topic.topic_type,
        "pipeline": pipeline,
        "condition": "full",
        "generator": generator,
        "patient_id": topic.patient_id,
        "type_question": topic.t3_query(),
        "note": topic.note,
        "system_answered": answered,
        "answer": answer,
        "packed_snippets": [
            {
                "doc_id": item.get("doc_id"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text") or item.get("content") or "",
                "source_type": item.get("source_type") or item.get("source"),
            }
            for item in passages
        ],
        "task_a": {
            "instruction": "Should a prudent CDS answer or abstain given only this packed evidence?",
            "show_answer": False,
            "show_system_name": False,
        },
        "task_b": claims,
        "raw_policy": response.get("policy_decision"),
        "raw_sufficiency": response.get("sufficiency_decision"),
        "trace_id": response.get("trace_id"),
    }


async def persist_packs(
    *,
    base_url: str,
    org_id: str,
    pipelines: list[str],
    generator: str,
    out_dir: Path,
    condition: str = "full",
) -> list[Path]:
    topics = load_topics()
    client_api = ProductClient(base_url, org_id, "t3-rater")
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as http:
        for topic in topics:
            await client_api.ingest_note(http, topic)
            for pipeline in pipelines:
                response = await client_api.chat(http, topic, pipeline, condition=condition)
                pack = build_pack(topic, pipeline, generator, response)
                pack["condition"] = condition
                path = out_dir / f"t3_{pipeline}_{generator}_{condition}_{topic.number}.json"
                path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
                written.append(path)
                print(f"wrote {path}", flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MEDSWIN_BASE_URL", "http://127.0.0.1:8110"))
    parser.add_argument("--org-id", default=os.environ.get("BENCHMARK_ORG_ID", "bench-org"))
    parser.add_argument("--pipeline", default="both", choices=("naive", "medswin", "both"))
    parser.add_argument("--generator", default="cloud", choices=("cloud", "medswin"))
    parser.add_argument("--condition", default="full", choices=("full", "no_gate", "no_mac"))
    parser.add_argument("--allow-local-t3", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=PACKS_DIR)
    args = parser.parse_args()
    if args.generator == "medswin" and not args.allow_local_t3:
        raise SystemExit("Human T3 freezes Foundry GPT. Pass --allow-local-t3 only for exploratory packs.")
    pipelines = ["naive", "medswin"] if args.pipeline == "both" else [args.pipeline]
    asyncio.run(
        persist_packs(
            base_url=args.base_url,
            org_id=args.org_id,
            pipelines=pipelines,
            generator=args.generator,
            out_dir=args.out_dir,
            condition=args.condition,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
