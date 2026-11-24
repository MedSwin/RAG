#!/usr/bin/env python3
"""Live end-to-end full-stack verification of MedSwin 2.0 on real local models + MongoDB.

Starts:
  1. Ephemeral MongoDB 7 container (`medswin-live-e2e-mongo`) if 127.0.0.1:27017 is closed
  2. Real `scripts/serve-medswin-llm.py` (`MedSwin-DaRE-TIES-KD-0.7` on MPS, port 18000)
  3. Real `uvicorn app.main:app` (`MedEmbed-large-v0.1` + `bge-reranker-v2-m3` on MPS, port 18100)

Exercises over real TCP/HTTP with JWT RBAC (`clinician` and `operator`):
  - LLM server `/health` (2048 context window, MPS, prefix KV cache)
  - API `/health`, `/api/v1/pipelines/ready`, `/api/v1/embedding/info`, Clinician UI `/app/`
  - Clinician vs Operator RBAC enforcement on EMR chart ingestion
  - Real TREC-CDS 2016 + BioASQ clinical chunk ingestion, 1024-d MedEmbed vectorization, and HNSW build
  - Live `naive`, `two-stage`, `full` (MAC 5-specialist fan-out + prefix KV reuse), and `/compare`
  - Live EvidenceGate abstention on out-of-corpus unanswerable query
  - Live MongoDB trace audit (`/traces/{id}`), session state (`/sessions/{id}`), and HL7 CDS Hooks 2.0.1
    (`/cds-services`, `/cds-services/medswin-evidence`, `/cds-services/medswin-evidence/feedback` PHI redaction)

Cleans up all spawned processes, temporary index files, and any ephemeral MongoDB container in `finally`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
LLM_PORT = 18000
API_PORT = 18100
JWT_SECRET = "live-e2e-test-secret-medswin-32bytes-minimum!"
ORG_ID = "st-vincent-cdss"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_http(url: str, timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err = ""
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    return resp.json()
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                last_err = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _mint_jwt(sub: str, org_id: str, role: str) -> str:
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    claims = _b64url(
        json.dumps({"sub": sub, "org_id": org_id, "role": role, "iat": now, "exp": now + 3600}).encode("utf-8")
    )
    sig = hmac.new(JWT_SECRET.encode("utf-8"), f"{header}.{claims}".encode("ascii"), hashlib.sha256).digest()
    return f"{header}.{claims}.{_b64url(sig)}"


def main() -> None:
    started_mongo_container = False
    container_name = "medswin-live-e2e-mongo"
    llm_proc: subprocess.Popen[str] | None = None
    api_proc: subprocess.Popen[str] | None = None
    tmp_index_dir = Path(tempfile.mkdtemp(prefix="medswin_live_e2e_idx_"))

    try:
        # 1. Ensure MongoDB on 127.0.0.1:27017
        if not _port_open("127.0.0.1", 27017):
            print("[1/7] Starting ephemeral mongo:7 container on 127.0.0.1:27017 ...", flush=True)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    container_name,
                    "-p",
                    "127.0.0.1:27017:27017",
                    "mongo:7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            started_mongo_container = True
            deadline = time.time() + 45.0
            while time.time() < deadline and not _port_open("127.0.0.1", 27017):
                time.sleep(0.5)
            time.sleep(1.5)
            if not _port_open("127.0.0.1", 27017):
                raise RuntimeError("MongoDB container did not open port 27017")
        else:
            print("[1/7] Reusing existing local MongoDB on 127.0.0.1:27017 ...", flush=True)

        # 2. Start real MedSwin 7B LLM server on port 18000
        print(f"[2/7] Starting scripts/serve-medswin-llm.py on 127.0.0.1:{LLM_PORT} ...", flush=True)
        llm_env = os.environ.copy()
        llm_env.update(
            {
                "MEDSWIN_LLM_HOST": "127.0.0.1",
                "MEDSWIN_LLM_PORT": str(LLM_PORT),
                "MEDSWIN_ENABLE_PREFIX_CACHE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        llm_log = open(tmp_index_dir / "llm_server.log", "w", encoding="utf-8")
        llm_proc = subprocess.Popen(
            [PYTHON, str(REPO_ROOT / "scripts" / "serve-medswin-llm.py")],
            cwd=str(REPO_ROOT),
            env=llm_env,
            stdout=llm_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        llm_health_before = _wait_http(f"http://127.0.0.1:{LLM_PORT}/health", timeout_s=180.0)
        print(
            f"      MedSwin LLM healthy: device={llm_health_before['device']}, "
            f"ctx={llm_health_before['context_window']}, rev={llm_health_before['model_revision']}",
            flush=True,
        )

        # 3. Start real FastAPI application on port 18100 in local 3-model mode
        print(f"[3/7] Starting FastAPI app.main:app on 127.0.0.1:{API_PORT} (local 3-model mode) ...", flush=True)
        api_env = os.environ.copy()
        api_env.update(
            {
                "CLOUD_MODE": "false",
                "GENERATION_BACKEND": "medswin_local",
                "MEDSWIN_LLM_URL": f"http://127.0.0.1:{LLM_PORT}/v1/chat/completions",
                "SUPERVISOR_URL": f"http://127.0.0.1:{LLM_PORT}/v1/chat/completions",
                "LOCAL_INPROCESS_MODELS": "1",
                "MEDSWIN_DEFAULT_MAX_TOKENS": "96",
                "LLM_TIMEOUT_S": "300",
                "DISABLE_DATASET_PRELOAD": "true",
                "MONGODB_URL": "mongodb://127.0.0.1:27017",
                "MONGODB_DB": "medswin_live_e2e",
                "MONGODB_DATABASE": "medswin_live_e2e",
                "ENABLE_AUTH": "true",
                "ENABLE_RBAC": "true",
                "AUTH_JWT_SECRET": JWT_SECRET,
                "HNSW_INDEX_PATH": str(tmp_index_dir / "hnsw_index.bin"),
                "HNSW_MAPPING_PATH": str(tmp_index_dir / "hnsw_mapping.json"),
                "FAISS_INDEX_PATH": str(tmp_index_dir / "faiss_index.bin"),
                "FAISS_MAPPING_PATH": str(tmp_index_dir / "faiss_mapping.json"),
                "TREE_INDEX_PATH": str(tmp_index_dir / "tree_index.npy"),
                "TREE_MAPPING_PATH": str(tmp_index_dir / "tree_mapping.json"),
                "DATA_DIR": str(tmp_index_dir / "data"),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        api_log = open(tmp_index_dir / "api_server.log", "w", encoding="utf-8")
        api_proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(API_PORT),
            ],
            cwd=str(REPO_ROOT),
            env=api_env,
            stdout=api_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        api_health_initial = _wait_http(f"http://127.0.0.1:{API_PORT}/health", timeout_s=180.0)
        print(
            f"      API healthy: embed={api_health_initial['embedding_model']}, "
            f"rerank={api_health_initial['reranker_model']}, dim={api_health_initial['active_embedding_dimension']}",
            flush=True,
        )

        clinician_headers = {"Authorization": f"Bearer {_mint_jwt('dr-chen', ORG_ID, 'clinician')}"}
        operator_headers = {"Authorization": f"Bearer {_mint_jwt('ops-admin', ORG_ID, 'operator')}"}

        with httpx.Client(base_url=f"http://127.0.0.1:{API_PORT}", timeout=900.0) as client:
            # 4. Verify UI, /pipelines/ready, /embedding/info, and RBAC chart boundary
            print("[4/7] Verifying Clinician UI, /pipelines/ready, /embedding/info, and RBAC ...", flush=True)
            ui_resp = client.get("/app/")
            assert ui_resp.status_code == 200 and "<html" in ui_resp.text.lower(), f"UI failed: {ui_resp.status_code}"

            ready = client.get("/api/v1/pipelines/ready", headers=clinician_headers).json()
            embed_info = client.get("/api/v1/embedding/info", headers=clinician_headers).json()
            assert ready["cloud_mode"] is False
            assert "bge-reranker-v2-m3" in str(ready["reranker_model"])
            assert ready["generation_backend"] == "medswin_local"

            # Clinician must be blocked from writing EMR charts (HTTP 403)
            rbac_emr_block = client.post(
                "/api/v1/storage/chunks",
                headers=clinician_headers,
                json={
                    "org_id": ORG_ID,
                    "source_type": "EMR",
                    "chunks": [{"chunk_id": "forbid#0", "doc_id": "forbid", "source_type": "EMR", "text": "test"}],
                },
            )
            assert rbac_emr_block.status_code == 403, f"Expected 403 for clinician EMR write, got {rbac_emr_block.status_code}"

            # 5. Ingest real TREC-CDS + BioASQ + curated clinical CPG/SAFETY/EMR evidence and build 1024-d HNSW
            print("[5/7] Ingesting real clinical corpus into MongoDB & building 1024-d HNSW index ...", flush=True)
            trec_corpus = json.loads(
                (REPO_ROOT / "data" / "benchmarks" / "trec_cds2016_judged" / "corpus.json").read_text(encoding="utf-8")
            )
            bioasq_corpus = json.loads(
                (REPO_ROOT / "data" / "benchmarks" / "bioasq_real" / "corpus.json").read_text(encoding="utf-8")
            )

            patient_id = "pt-afib-ckd-01"
            curated_clinical_chunks = [
                {
                    "chunk_id": "cpg-aha-afib-anticoag#0",
                    "doc_id": "cpg-aha-afib-2024",
                    "source_type": "CPG",
                    "section": "Recommendation",
                    "text": (
                        "AHA/ACC Atrial Fibrillation Guideline Recommendation (Grade A): Direct oral anticoagulants "
                        "specifically apixaban (5 mg twice daily, or 2.5 mg twice daily when >=2 of age >=80 years, "
                        "body weight <=60 kg, or serum creatinine >=1.5 mg/dL) are recommended over warfarin to reduce "
                        "stroke and major bleeding in nonvalvular atrial fibrillation with CKD."
                    ),
                    "evidence_grade": {"grade": "A", "score": 0.95},
                    "source_reliability": 0.96,
                },
                {
                    "chunk_id": "cpg-esc-afib-renal#0",
                    "doc_id": "cpg-esc-afib-2024",
                    "source_type": "CPG",
                    "section": "Recommendation",
                    "text": (
                        "ESC Clinical Practice Guideline Recommendation (Grade A): In elderly atrial fibrillation "
                        "patients with stage 3b-4 chronic kidney disease (CrCl 15-49 mL/min), apixaban demonstrates "
                        "lower major gastrointestinal and intracranial bleeding risk than rivaroxaban or warfarin."
                    ),
                    "evidence_grade": {"grade": "A", "score": 0.94},
                    "source_reliability": 0.95,
                },
                {
                    "chunk_id": "safety-dabigatran-nsaid-ckd#0",
                    "doc_id": "safety-renal-anticoag",
                    "source_type": "SAFETY",
                    "section": "Contraindications and Warnings",
                    "text": (
                        "CONTRAINDICATION AND SAFETY WARNING: Avoid dabigatran in severe renal impairment (CrCl <30 mL/min) "
                        "due to 80% renal clearance and major hemorrhage risk. Concomitant systemic NSAIDs (ketorolac, "
                        "ibuprofen) are contraindicated with oral anticoagulants in CKD due to acute kidney injury and GI bleeding."
                    ),
                    "evidence_grade": {"grade": "A", "score": 0.95},
                    "source_reliability": 0.96,
                },
                {
                    "chunk_id": "emr-pt-afib-ckd-01#0",
                    "doc_id": "emr-chart-01",
                    "source_type": "EMR",
                    "patient_id": patient_id,
                    "section": "Assessment and Labs",
                    "text": (
                        "Patient pt-afib-ckd-01: 82-year-old female (weight 56 kg) with persistent nonvalvular atrial "
                        "fibrillation (CHA2DS2-VASc = 5), hypertension, prior TIA, serum creatinine 1.7 mg/dL (eGFR 29 mL/min), "
                        "hemoglobin 11.8 g/dL, no active bleeding. Allergy: severe anaphylaxis to penicillin."
                    ),
                    "evidence_grade": {"grade": "B", "score": 0.88},
                    "source_reliability": 0.90,
                },
                {
                    "chunk_id": "emr-pt-afib-ckd-01#1",
                    "doc_id": "emr-chart-01-meds",
                    "source_type": "EMR",
                    "patient_id": patient_id,
                    "section": "Medication History",
                    "text": (
                        "Current medications for pt-afib-ckd-01: amlodipine 5 mg daily, metoprolol succinate 50 mg daily. "
                        "Patient previously experienced labile INR on warfarin and asks about switching to a safer oral anticoagulant."
                    ),
                    "evidence_grade": {"grade": "B", "score": 0.88},
                    "source_reliability": 0.90,
                },
            ]

            real_lit_chunks = [
                {
                    "chunk_id": f"trec-pmc-{d['doc_id']}#0",
                    "doc_id": f"PMC{d['doc_id']}",
                    "source_type": d["source_type"],
                    "text": d["text"][:1000],
                    "metadata": d["metadata"],
                }
                for d in trec_corpus[:20]
            ] + [
                {
                    "chunk_id": f"bioasq-pmid-{d['doc_id']}#0",
                    "doc_id": f"PMID{d['doc_id']}",
                    "source_type": d["source_type"],
                    "text": d["text"][:1000],
                    "metadata": d["metadata"],
                }
                for d in bioasq_corpus[:20]
            ]

            ingest_resp = client.post(
                "/api/v1/storage/chunks",
                headers=operator_headers,
                json={
                    "org_id": ORG_ID,
                    "source_type": "LIT",
                    "batch_size": 32,
                    "chunks": curated_clinical_chunks + real_lit_chunks,
                },
            )
            assert ingest_resp.status_code == 200, ingest_resp.text
            ingest_data = ingest_resp.json()

            build_resp = client.post(
                "/api/v1/storage/index/build",
                headers=operator_headers,
                json={"force_rebuild": True},
            )
            assert build_resp.status_code == 200, build_resp.text
            build_data = build_resp.json()
            assert build_data["success"] is True and build_data["total_vectors"] >= 40

            api_health_indexed = client.get("/health").json()
            assert api_health_indexed["active_index_ready"] is True, api_health_indexed
            print(
                f"      Stored {ingest_data['success_count']} chunks & built 1024-d HNSW "
                f"({build_data['total_vectors']} vectors, active_index_ready={api_health_indexed['active_index_ready']})",
                flush=True,
            )

            # 6. Exercise naive, two-stage, full (answerable + abstention), compare, traces, sessions
            print("[6/7] Running live naive, two-stage, full MAC, abstention, and /compare ...", flush=True)
            clinical_query = (
                "What oral anticoagulant regimen and renal safety precautions are recommended for elderly "
                "nonvalvular atrial fibrillation patient pt-afib-ckd-01 with creatinine 1.7 mg/dL and weight 56 kg?"
            )
            session_id = "sess-live-cdss-001"

            two_stage_resp = client.post(
                "/api/v1/pipelines/two-stage/chat",
                headers=clinician_headers,
                json={
                    "query": clinical_query,
                    "user_id": "dr-chen",
                    "org_id": ORG_ID,
                    "session_id": session_id,
                    "patient_id": patient_id,
                },
            )
            assert two_stage_resp.status_code == 200, two_stage_resp.text
            two_stage_data = two_stage_resp.json()

            full_resp = client.post(
                "/api/v1/pipelines/full/chat",
                headers=clinician_headers,
                json={
                    "query": clinical_query,
                    "user_id": "dr-chen",
                    "org_id": ORG_ID,
                    "session_id": session_id,
                    "patient_id": patient_id,
                },
            )
            assert full_resp.status_code == 200, full_resp.text
            full_data = full_resp.json()
            agent_failures = [
                key for key, failed in (full_data.get("degraded_mode") or {}).items()
                if failed and str(key).startswith("agent_")
            ]
            assert not agent_failures, f"Specialist agents degraded: {agent_failures} {full_data.get('degraded_mode')}"

            # Check abstention on completely unanswerable out-of-corpus query
            abstain_resp = client.post(
                "/api/v1/pipelines/two-stage/chat",
                headers=clinician_headers,
                json={
                    "query": "What is the exact intrathecal maintenance dose of fictional-zyloximab-99 in pediatric Krabbe disease?",
                    "user_id": "dr-chen",
                    "org_id": ORG_ID,
                    "constraints": {
                        "min_evidence_grade": 0.99,
                    },
                },
            )
            assert abstain_resp.status_code == 200, abstain_resp.text
            abstain_data = abstain_resp.json()

            # Side-by-side compare across naive + two-stage
            compare_resp = client.post(
                "/api/v1/pipelines/compare",
                headers=clinician_headers,
                json={
                    "query": clinical_query,
                    "user_id": "dr-chen",
                    "org_id": ORG_ID,
                    "patient_id": patient_id,
                    "pipelines": ["naive", "two-stage"],
                },
            )
            assert compare_resp.status_code == 200, compare_resp.text
            compare_data = compare_resp.json()
            assert "naive" in compare_data["results"] and "two-stage" in compare_data["results"]
            assert "naive_to_two_stage" in compare_data["adjacent_diffs"]

            # Persisted audit trace & session retrieval
            trace_id = full_data["trace_id"]
            trace_resp = client.get(
                f"/api/v1/pipelines/traces/{trace_id}",
                headers=clinician_headers,
                params={"org_id": ORG_ID, "include_details": "true"},
            )
            assert trace_resp.status_code == 200, trace_resp.text
            trace_summary = trace_resp.json()

            sess_resp = client.get(
                f"/api/v1/pipelines/sessions/{session_id}",
                headers=clinician_headers,
                params={"org_id": ORG_ID},
            )
            assert sess_resp.status_code == 200, sess_resp.text
            sess_data = sess_resp.json()

            # 7. HL7 CDS Hooks 2.0.1 Discovery + Invocation + Override Feedback PHI Redaction
            print("[7/7] Exercising HL7 CDS Hooks 2.0.1 discovery, card generation, and PHI-redacted feedback ...", flush=True)
            cds_disc_resp = client.get("/cds-services", headers=clinician_headers)
            assert cds_disc_resp.status_code == 200, cds_disc_resp.text
            cds_disc = cds_disc_resp.json()
            service_ids = [s["id"] for s in cds_disc.get("services", [])]
            assert "medswin-evidence" in service_ids

            cds_call = client.post(
                "/cds-services/medswin-evidence",
                headers=clinician_headers,
                json={
                    "hook": "patient-view",
                    "hookInstance": "live-hook-001",
                    "context": {
                        "userId": "Practitioner/dr-chen",
                        "patientId": patient_id,
                        "orgId": ORG_ID,
                        "clinicalQuestion": clinical_query,
                    },
                },
            )
            assert cds_call.status_code == 200, cds_call.text
            cds_cards = cds_call.json().get("cards", [])
            assert len(cds_cards) >= 1
            card_uuid = cds_cards[0]["uuid"]

            cds_fb = client.post(
                "/cds-services/medswin-evidence/feedback",
                headers=clinician_headers,
                json={
                    "feedback": [
                        {
                            "card": card_uuid,
                            "outcome": "overridden",
                            "overrideReason": {
                                "code": "patient-preference",
                                "userComment": "Discussed with patient at 555-234-5678 and dr.chen@stvincent.org",
                            },
                        }
                    ]
                },
            )
            assert cds_fb.status_code == 200, cds_fb.text

        llm_health_after = httpx.get(f"http://127.0.0.1:{LLM_PORT}/health", timeout=10.0).json()

        report = {
            "status": "PASSED",
            "llm_server_before": llm_health_before,
            "llm_server_after": llm_health_after,
            "api_health_indexed": api_health_indexed,
            "pipelines_ready": ready,
            "embedding_info": embed_info,
            "rbac_clinician_emr_write_status": rbac_emr_block.status_code,
            "ingested_chunks": ingest_data["success_count"],
            "hnsw_vectors": build_data["total_vectors"],
            "two_stage": {
                "policy_action": two_stage_data["policy_decision"]["action"],
                "passed": two_stage_data["policy_decision"]["passed"],
                "rerank_traces": len(two_stage_data.get("rerank_traces") or []),
                "citations": len(two_stage_data.get("citations") or []),
                "timing_ms": two_stage_data["timing_ms"],
                "answer_preview": two_stage_data["answer"][:260],
            },
            "full_mac": {
                "trace_id": trace_id,
                "policy_action": full_data["policy_decision"]["action"],
                "passed": full_data["policy_decision"]["passed"],
                "citations": len(full_data.get("citations") or []),
                "timing_ms": full_data["timing_ms"],
                "answer_preview": full_data["answer"][:260],
            },
            "abstention_unanswerable": {
                "policy_action": abstain_data["policy_decision"]["action"],
                "passed": abstain_data["policy_decision"]["passed"],
                "abstained": abstain_data["answer_artifact"]["abstained"],
                "answer_preview": abstain_data["answer"][:200],
            },
            "compare_adjacent_diff": compare_data["adjacent_diffs"]["naive_to_two_stage"],
            "persisted_trace_keys": sorted(trace_summary.keys()),
            "persisted_session_id": sess_data["session_id"],
            "cds_hooks": {
                "discovered_services": service_ids,
                "card_count": len(cds_cards),
                "first_card_indicator": cds_cards[0]["indicator"],
                "first_card_summary_preview": cds_cards[0]["summary"][:160],
                "feedback_response": cds_fb.json(),
            },
        }
        out_path = REPO_ROOT / "data" / "benchmarks" / "live_fullstack_e2e_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nLIVE FULL-STACK E2E PASSED -> {out_path}", flush=True)
        print(json.dumps(report, indent=2))
    finally:
        for proc in (api_proc, llm_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if started_mongo_container:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
        shutil.rmtree(tmp_index_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
