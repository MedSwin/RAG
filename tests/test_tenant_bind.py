import asyncio
import base64
import hashlib
import hmac
import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import AuthMiddleware
from app.api.cds import get_cds_chat, router as cds_router
from app.api.tenant import bind_org, bind_tenant, require_operator
from app.api.v1.endpoints import dashboard, embedding, pipelines, preprocessing, retrieval, storage
from app.services.ingestion import IngestionPipelineService
from app.core.config import settings
from app.services import storage as storage_service
from app.schemas.enums import PipelineId
from app.schemas.evidence import EvidenceBundle
from app.schemas.traces import ChatResponse


class _State:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _Request:
    def __init__(self, **state):
        self.state = _State(**state)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _token(secret: str, claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(signature)}"


def test_a_trace_read_hides_another_clinician_in_the_same_org(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "AUTH_JWT_AUDIENCE", None)
    monkeypatch.setattr(settings, "AUTH_JWT_ISSUER", None)

    class _Traces:
        async def get_by_id(self, trace_id, org_id):
            return {
                "trace_id": trace_id,
                "session_id": "sess-1",
                "user_id": "dr-chen",
                "query": "Which anticoagulant is safer for this patient?",
                "messages": [],
                "tool_calls": [],
                "sufficiency_checks": [],
                "evidence_bundle": {"passages": []},
            }

    monkeypatch.setattr(pipelines, "TraceRepository", lambda: _Traces())
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(pipelines.router, prefix="/pipelines")
    client = TestClient(app)
    other = _token(secret, {"sub": "dr-other", "org_id": "org-a", "exp": int(time.time()) + 60})
    owner = _token(secret, {"sub": "dr-chen", "org_id": "org-a", "exp": int(time.time()) + 60})
    hidden = client.get(
        "/pipelines/traces/trace-1",
        headers={"Authorization": f"Bearer {other}"},
        params={"org_id": "org-a", "include_details": "true"},
    )
    shown = client.get(
        "/pipelines/traces/trace-1",
        headers={"Authorization": f"Bearer {owner}"},
        params={"org_id": "org-a"},
    )
    assert hidden.status_code == 404
    assert "anticoagulant" not in hidden.text
    assert shown.status_code == 200
    assert shown.json()["query"].startswith("Which anticoagulant")


def test_a_session_read_hides_another_clinician_in_the_same_org(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "AUTH_JWT_AUDIENCE", None)
    monkeypatch.setattr(settings, "AUTH_JWT_ISSUER", None)

    class _Sessions:
        async def get_by_id(self, session_id, org_id):
            return {
                "user_id": "dr-chen",
                "created_at": "t0",
                "last_active": "t1",
                "metadata": {"note": "private", "phone": "Call 555-234-5678"},
            }

    monkeypatch.setattr(pipelines, "SessionRepository", lambda: _Sessions())
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(pipelines.router, prefix="/pipelines")
    client = TestClient(app)
    other = _token(secret, {"sub": "dr-other", "org_id": "org-a", "exp": int(time.time()) + 60})
    owner = _token(secret, {"sub": "dr-chen", "org_id": "org-a", "exp": int(time.time()) + 60})
    hidden = client.get(
        "/pipelines/sessions/sess-1",
        headers={"Authorization": f"Bearer {other}"},
        params={"org_id": "org-a"},
    )
    shown = client.get(
        "/pipelines/sessions/sess-1",
        headers={"Authorization": f"Bearer {owner}"},
        params={"org_id": "org-a"},
    )
    assert hidden.status_code == 404
    assert "private" not in hidden.text
    assert shown.status_code == 200
    assert shown.json()["user_id"] == "dr-chen"
    assert shown.json()["metadata"]["note"] == "private"
    assert "555-234-5678" not in shown.text


def test_unsigned_request_keeps_the_body_organisation():
    user_id, org_id = bind_tenant(_Request(), "clinician-1", "org-a")
    assert (user_id, org_id) == ("clinician-1", "org-a")
    assert bind_org(_Request(), "org-a") == "org-a"


def test_token_rejects_a_different_organisation():
    request = _Request(user_id="clinician-1", org_id="org-a")
    try:
        bind_tenant(request, "clinician-1", "org-b")
    except Exception as exc:
        assert exc.status_code == 403
        assert "organisation" in exc.detail
    else:
        raise AssertionError("A different organisation was accepted")


def test_a_fhir_reference_matches_the_stored_id():
    request = _Request(user_id="Practitioner/dr-chen", org_id="org-a")
    assert bind_tenant(request, "Practitioner/dr-chen", "org-a") == ("dr-chen", "org-a")


def test_token_fills_a_missing_organisation():
    request = _Request(user_id="clinician-1", org_id="org-a")
    assert bind_tenant(request, "", "") == ("clinician-1", "org-a")


def test_cds_token_blocks_another_organisation(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        raise AssertionError("chat should not run")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(cds_router)
    app.dependency_overrides[get_cds_chat] = lambda: chat
    token = _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})
    response = TestClient(app).post(
        "/cds-services/medswin-evidence",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "hook": "patient-view",
            "hookInstance": "hook-1",
            "context": {"question": "What is the first-line antibiotic?", "orgId": "org-b", "userId": "clinician-1"},
        },
    )
    assert response.status_code == 403
    assert calls == []


def test_pipeline_token_uses_the_token_organisation(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    seen = []

    async def _chat(pipeline, request):
        seen.append((pipeline, request.user_id, request.org_id))
        return ChatResponse(
            answer="held",
            evidence_bundle=EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0),
            trace_id="trace-1",
            pipeline=pipeline,
        )

    monkeypatch.setattr(pipelines, "_chat", _chat)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")
    token = _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})
    response = TestClient(app).post(
        "/api/v1/pipelines/two-stage/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "q", "user_id": "clinician-1", "org_id": "org-a"},
    )
    assert response.status_code == 200
    assert seen == [(PipelineId.TWO_STAGE, "clinician-1", "org-a")]

    denied = TestClient(app).post(
        "/api/v1/pipelines/two-stage/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "q", "user_id": "clinician-1", "org_id": "org-b"},
    )
    assert denied.status_code == 403
    assert seen == [(PipelineId.TWO_STAGE, "clinician-1", "org-a")]


def test_a_service_error_hides_the_exception_text(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)
    monkeypatch.setattr(settings, "CLOUD_MODE", True)

    async def _cloud_embed(texts, input_type):
        raise RuntimeError("vector 555-234-5678")

    class _Chunker:
        async def chunk_medical_dialogues(self, frame, target_chunk_size):
            raise RuntimeError("note 555-234-5678")

    class _Datasets:
        async def get_dataset_info(self, name):
            raise RuntimeError("repo 555-234-5678")

    class _Path:
        def __init__(self, *_args, **_kwargs):
            return None

        def exists(self):
            return True

    class _Builder:
        def __init__(self, _dimension):
            self.mapping = type("Mapping", (), {"close": staticmethod(lambda: None)})()

        def load(self, *_args):
            raise RuntimeError("index 555-234-5678")

    monkeypatch.setattr(embedding, "_cloud_embed", _cloud_embed)
    monkeypatch.setattr(preprocessing, "get_preprocessing_service", lambda: _Chunker())
    monkeypatch.setattr(retrieval, "Path", _Path)
    monkeypatch.setattr(retrieval, "HNSWIndexBuilder", _Builder)
    app = FastAPI()
    app.include_router(embedding.router, prefix="/api/v1/embedding")
    app.include_router(preprocessing.router, prefix="/api/v1/preprocessing")
    app.include_router(dashboard.router, prefix="/api/v1/dashboard")
    app.include_router(retrieval.router, prefix="/api/v1/retrieval")
    app.dependency_overrides[dashboard.get_hf_service] = lambda: _Datasets()
    client = TestClient(app)
    embedded = client.post("/api/v1/embedding/embed", json={"text": "dose"})
    chunked = client.post("/api/v1/preprocessing/chunk", json={"data": [{"text": "dose"}]})
    listed = client.get("/api/v1/dashboard/dataset/medqa")
    indexed = client.get("/api/v1/retrieval/index/info")
    assert embedded.status_code == 500
    assert embedded.json()["detail"] == "Embedding failed: RuntimeError"
    assert chunked.status_code == 500
    assert chunked.json()["detail"] == "Chunking failed: RuntimeError"
    assert listed.status_code == 500
    assert listed.json()["detail"] == "Failed to get dataset details: RuntimeError"
    assert indexed.status_code == 500
    assert indexed.json()["detail"] == "Failed to get index info: RuntimeError"
    assert "555-234-5678" not in embedded.text + chunked.text + listed.text + indexed.text


def test_a_storage_error_hides_the_exception_text(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)

    class Storage:
        async def list_chunks(self, skip, limit, filters):
            raise RuntimeError("chart row 555-234-5678")

    app = FastAPI()
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    response = TestClient(app).get("/api/v1/storage/chunks", params={"org_id": "org-a"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to list chunks: RuntimeError"
    assert "555-234-5678" not in response.text


def test_compare_keeps_a_successful_arm_when_one_pipeline_fails(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)

    async def _chat(pipeline, request):
        if pipeline == PipelineId.NAIVE:
            raise RuntimeError("patient phone 555-234-5678")
        return ChatResponse(
            answer="held",
            evidence_bundle=EvidenceBundle(passages=[], total_tokens=0, cpg_count=0, emr_count=0, lit_count=0),
            trace_id="trace-full",
            pipeline=pipeline,
        )

    monkeypatch.setattr(pipelines, "_chat", _chat)
    app = FastAPI()
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")
    response = TestClient(app).post(
        "/api/v1/pipelines/compare",
        json={
            "query": "Call 555-234-5678 about metformin",
            "user_id": "clinician-1",
            "org_id": "org-a",
            "pipelines": ["naive", "full"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "555-234-5678" not in response.text
    assert body["query"] == "Call [REDACTED_PHONE] about metformin"
    assert body["results"]["full"]["answer"] == "held"
    failed = body["results"]["naive"]
    assert failed["degraded_mode"]["error"] is True
    assert failed["safety_notes"] == "RuntimeError"
    assert failed["policy_decision"]["reason"] == "naive pipeline failed: RuntimeError"
    assert failed["answer_artifact"]["abstained"] is True


def test_a_pipeline_error_hides_the_exception_text(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_AUTH", False)

    async def _chat(pipeline, request):
        raise RuntimeError("patient phone 555-234-5678")

    async def _search(request):
        raise RuntimeError("chart text 555-234-5678")

    monkeypatch.setattr(pipelines, "_chat", _chat)
    monkeypatch.setattr(retrieval, "_search", _search)
    app = FastAPI()
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")
    app.include_router(retrieval.router, prefix="/api/v1/retrieval")
    client = TestClient(app)
    chat = client.post(
        "/api/v1/pipelines/full/chat",
        json={"query": "q", "user_id": "clinician-1", "org_id": "org-a"},
    )
    search = client.post(
        "/api/v1/retrieval/search",
        json={"query": "q", "org_id": "org-a"},
    )
    assert chat.status_code == 500
    assert chat.json()["detail"] == "full pipeline failed: RuntimeError"
    assert "555-234-5678" not in chat.text
    assert search.status_code == 500
    assert search.json()["detail"] == "Search failed: RuntimeError"
    assert "555-234-5678" not in search.text


def _auth(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    return _token(secret, {"sub": "clinician-1", "org_id": "org-a", "exp": int(time.time()) + 60})


def test_storage_token_blocks_another_organisation(monkeypatch):
    token = _auth(monkeypatch)
    listed = []

    class Storage:
        async def list_chunks(self, skip, limit, filters):
            listed.append(filters["org_id"])
            return []

        async def clear_benchmark_org(self, org_id, remove_indexes):
            raise AssertionError("reset should not run")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    denied = client.get("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-b"})
    assert denied.status_code == 403
    assert listed == []
    allowed = client.get("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-a"})
    assert allowed.status_code == 200
    assert listed == ["org-a"]
    reset = client.post(
        "/api/v1/storage/benchmark/reset",
        headers=headers,
        json={"org_id": "org-b"},
    )
    assert reset.status_code == 403


def test_retrieval_token_blocks_another_organisation_before_embed(monkeypatch):
    token = _auth(monkeypatch)

    async def _search(request):
        raise AssertionError("search should not run")

    monkeypatch.setattr(retrieval, "_search", _search)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(retrieval.router, prefix="/api/v1/retrieval")
    response = TestClient(app).post(
        "/api/v1/retrieval/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "first-line antibiotic", "org_id": "org-b", "use_reranking": False},
    )
    assert response.status_code == 403


def test_shared_index_build_requires_an_operator(monkeypatch):
    token = _auth(monkeypatch)
    built = []

    class Storage:
        async def build_hnsw_index_async(self, **kwargs):
            built.append(kwargs)
            return {
                "success": True,
                "index_path": "configured.bin",
                "mapping_path": "configured.json",
                "total_vectors": 0,
                "message": "built",
            }

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    client = TestClient(app)
    denied = client.post(
        "/api/v1/storage/index/build",
        headers={"Authorization": f"Bearer {token}"},
        json={"force_rebuild": True},
    )
    assert denied.status_code == 403
    assert built == []
    operator = _token(
        "clinical-secret-value",
        {"sub": "ops-1", "org_id": "org-a", "role": "operator", "exp": int(time.time()) + 60},
    )
    allowed = client.post(
        "/api/v1/storage/index/build",
        headers={"Authorization": f"Bearer {operator}"},
        json={"force_rebuild": True},
    )
    assert allowed.status_code == 200
    assert built[0]["index_path"] == settings.HNSW_INDEX_PATH
    assert built[0]["mapping_path"] == settings.HNSW_MAPPING_PATH
    rejected = client.post(
        "/api/v1/storage/index/build",
        headers={"Authorization": f"Bearer {operator}"},
        json={"index_path": "/tmp/other.bin"},
    )
    assert rejected.status_code == 400
    assert len(built) == 1
    scoped = client.post(
        "/api/v1/storage/index/build",
        headers={"Authorization": f"Bearer {operator}"},
        json={"force_rebuild": True, "org_id": "org-a"},
    )
    assert scoped.status_code == 400
    assert len(built) == 1


def test_operator_index_notice_names_the_shared_index():
    from app.cli.operator import shared_index_notice

    text = shared_index_notice("org-a")
    assert "org=org-a" in text
    assert "shared index" in text


def test_operator_can_build_the_index_when_rbac_is_on(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "RBAC_ALLOWED_ROLES", "clinician")

    class Storage:
        async def build_hnsw_index_async(self, **kwargs):
            return {
                "success": True,
                "index_path": kwargs["index_path"],
                "mapping_path": kwargs["mapping_path"],
                "total_vectors": 0,
                "message": "built",
            }

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    operator = _token(secret, {"sub": "ops-1", "org_id": "org-a", "role": "operator", "exp": int(time.time()) + 60})
    response = TestClient(app).post(
        "/api/v1/storage/index/build",
        headers={"Authorization": f"Bearer {operator}"},
        json={},
    )
    assert response.status_code == 200


def test_local_index_build_stays_open_without_a_token():
    require_operator(_Request())


def test_clinician_store_does_not_rebuild_the_shared_index(monkeypatch):
    token = _auth(monkeypatch)
    built = []

    class Storage:
        async def store_chunks(self, **kwargs):
            return {"success_count": 1, "skipped_count": 0, "failed_count": 0}

        async def build_hnsw_index_async(self, **kwargs):
            built.append(kwargs)
            return {"success": True}

    async def _attach(chunks, batch_size):
        return None

    monkeypatch.setattr(storage, "_attach_missing_embeddings", _attach)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    response = TestClient(app).post(
        "/api/v1/storage/chunks",
        headers={"Authorization": f"Bearer {token}"},
        json={"org_id": "org-a", "chunks": [{"chunk_id": "c1", "doc_id": "d1", "text": "Metformin."}]},
    )
    assert response.status_code == 200
    assert "was not rebuilt" in response.json()["message"]
    assert built == []


def test_clinician_cannot_store_a_chart(monkeypatch):
    token = _auth(monkeypatch)
    stored = []

    class Storage:
        async def store_chunks(self, **kwargs):
            stored.append(kwargs)
            return {"success_count": 1, "skipped_count": 0, "failed_count": 0}

    async def _attach(chunks, batch_size):
        raise AssertionError("embed should not run")

    monkeypatch.setattr(storage, "_attach_missing_embeddings", _attach)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    response = TestClient(app).post(
        "/api/v1/storage/chunks",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "org_id": "org-a",
            "chunks": [{"chunk_id": "c1", "doc_id": "d1", "text": "Patient chart.", "source_type": "EMR"}],
        },
    )
    assert response.status_code == 403
    assert stored == []


def test_operator_refresh_schedules_the_shared_index(monkeypatch):
    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", False)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "AUTH_JWT_ALGORITHM", "HS256")
    built = []
    charts = []

    class Storage:
        async def build_hnsw_index_async(self, **kwargs):
            built.append(kwargs)
            return {"success": True}

    async def _refresh(org_id, batch_size, *, include_charts=True):
        charts.append(include_charts)
        return {"org_id": org_id, "updated": 1}

    monkeypatch.setattr(storage, "_refresh_tenant_embeddings", _refresh)
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    clinician = _token(secret, {"sub": "clinician-1", "org_id": "org-a", "role": "clinician", "exp": int(time.time()) + 60})
    held = TestClient(app).post(
        "/api/v1/storage/embeddings/refresh",
        headers={"Authorization": f"Bearer {clinician}"},
        json={"org_id": "org-a"},
    )
    assert held.status_code == 200
    assert held.json()["index_rebuild"] == "operator_required"
    assert built == []
    assert charts == [False]
    operator = _token(secret, {"sub": "ops-1", "org_id": "org-a", "role": "operator", "exp": int(time.time()) + 60})
    allowed = TestClient(app).post(
        "/api/v1/storage/embeddings/refresh",
        headers={"Authorization": f"Bearer {operator}"},
        json={"org_id": "org-a"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["index_rebuild"] == "scheduled"
    assert charts == [False, True]
    assert built[0]["org_id"] is None
    assert built[0]["force_rebuild"] is True


def test_the_chunk_read_drops_a_phone_number():
    public = storage._public_chunk(
        {
            "chunk_id": "emr-1",
            "patient_id": "pt-1",
            "text": "Call 555-234-5678. Creatinine is 1.4. Continue metformin.",
            "metadata": {"patient_id": "pt-1", "note": "Call 555-234-5678."},
        }
    )
    assert "555-234-5678" not in public["text"]
    assert "Creatinine is 1.4." in public["text"]
    assert "metformin" in public["text"]
    assert public["patient_id"] == "pt-1"
    assert public["metadata"]["patient_id"] == "pt-1"
    assert "555-234-5678" not in public["metadata"]["note"]


def test_clinician_cannot_list_or_delete_charts(monkeypatch):
    token = _auth(monkeypatch)
    seen = []

    class Storage:
        async def list_chunks(self, skip, limit, filters):
            seen.append(filters)
            return [{"chunk_id": "guide", "source_type": "CPG"}]

        async def get_chunk(self, chunk_id, org_id):
            if chunk_id == "chart":
                return {"chunk_id": chunk_id, "source_type": "EMR", "text": "Patient chart."}
            return {"chunk_id": chunk_id, "source_type": "CPG", "text": "Guideline."}

        async def clear_chunks(self, collection_name, org_id):
            raise AssertionError("clear should not run")

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(storage.router, prefix="/api/v1/storage")
    app.dependency_overrides[storage.get_storage_service] = lambda: Storage()
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}
    listed = client.get("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-a"})
    assert listed.status_code == 200
    assert seen[0]["source_type"] == {"$ne": "EMR"}
    hidden = client.get("/api/v1/storage/chunks/chart", headers=headers, params={"org_id": "org-a"})
    assert hidden.status_code == 404
    visible = client.get("/api/v1/storage/chunks/guide", headers=headers, params={"org_id": "org-a"})
    assert visible.status_code == 200
    assert visible.json()["source_type"] == "CPG"
    denied = client.delete("/api/v1/storage/chunks", headers=headers, params={"org_id": "org-a"})
    assert denied.status_code == 403


def test_service_refresh_skips_charts_and_does_not_rebuild(monkeypatch):
    embedded = []
    built = []

    class _Cursor:
        def limit(self, _count):
            return self

        async def to_list(self, length):
            return []

    class _Collection:
        async def count_documents(self, query):
            assert query["org_id"] == "org-a"
            assert query["source_type"] == {"$ne": "EMR"}
            return 0

        def find(self, _query):
            return _Cursor()

    class _Client:
        def __init__(self, _url):
            pass

        async def embed(self, texts, **_kwargs):
            embedded.append(texts)
            return []

        async def close(self):
            return None

    service = storage_service.StorageService()
    previous = dict(storage_service.EMBEDDING_REFRESH_STATUS)

    async def _build(**kwargs):
        built.append(kwargs)

    monkeypatch.setattr(storage_service, "get_database", lambda: {"chunks": _Collection()})
    monkeypatch.setattr(storage_service, "EmbeddingClient", _Client)
    monkeypatch.setattr(service, "build_hnsw_index_async", _build)
    try:
        status = asyncio.run(service.refresh_cloud_embeddings(org_id="org-a"))
    finally:
        storage_service.EMBEDDING_REFRESH_STATUS.clear()
        storage_service.EMBEDDING_REFRESH_STATUS.update(previous)
    assert embedded == []
    assert built == []
    assert status["running"] is False
    assert status["error"] is None


def test_embedding_refresh_status_hides_the_exception_text(monkeypatch):
    previous = dict(storage_service.EMBEDDING_REFRESH_STATUS)

    def _boom():
        raise RuntimeError("secret connection 555-234-5678")

    monkeypatch.setattr(storage_service, "get_database", _boom)
    try:
        status = asyncio.run(storage_service.StorageService().refresh_cloud_embeddings(org_id="org-a"))
    finally:
        storage_service.EMBEDDING_REFRESH_STATUS.clear()
        storage_service.EMBEDDING_REFRESH_STATUS.update(previous)
    assert status["error"] == "RuntimeError"
    assert "555-234-5678" not in str(status)


def test_ingestion_status_hides_the_exception_text(monkeypatch):
    from app.services import ingestion as ingestion_module

    def _boom():
        raise RuntimeError("mongo uri 555-234-5678")

    monkeypatch.setattr(ingestion_module, "get_sync_database", _boom)
    service = ingestion_module.IngestionPipelineService()
    try:
        status = asyncio.run(service.get_ingestion_status("medqa"))
    finally:
        service.cleanup()
    assert status["error"] == "RuntimeError"
    assert status["status"] == "error"
    assert "555-234-5678" not in str(status)


def test_cloud_dataset_ingest_does_not_embed_or_rebuild(monkeypatch):
    monkeypatch.setattr(settings, "CLOUD_MODE", True)
    service = IngestionPipelineService()
    called = []

    async def _crawl(name):
        called.append(name)
        return {"success": True, "processed_data": []}

    async def _build(**kwargs):
        called.append(kwargs)

    service.hf_service.crawl_dataset = _crawl
    service.storage_service.build_hnsw_index_async = _build
    result = asyncio.run(service.run_full_ingestion_pipeline("meddialog"))
    assert result["success"] is False
    assert called == []

    secret = "clinical-secret-value"
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    monkeypatch.setattr(settings, "ENABLE_RBAC", True)
    monkeypatch.setattr(settings, "AUTH_JWT_SECRET", secret)
    monkeypatch.setattr(settings, "RBAC_ALLOWED_ROLES", "clinician")
    scheduled = []

    class Pipeline:
        async def run_full_ingestion_pipeline(self, name):
            scheduled.append(name)
            return {"success": True}

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(dashboard.router, prefix="/api/v1/dashboard")
    app.dependency_overrides[dashboard.get_ingestion_service] = lambda: Pipeline()
    operator = _token(secret, {"sub": "ops-1", "org_id": "org-a", "role": "operator", "exp": int(time.time()) + 60})
    denied = TestClient(app).post(
        "/api/v1/dashboard/ingest/meddialog",
        headers={"Authorization": f"Bearer {operator}"},
    )
    assert denied.status_code == 409
    assert scheduled == []

    monkeypatch.setattr(settings, "CLOUD_MODE", False)
    allowed = TestClient(app).post(
        "/api/v1/dashboard/ingest/meddialog",
        headers={"Authorization": f"Bearer {operator}"},
    )
    assert allowed.status_code == 200
    assert scheduled == ["meddialog"]
