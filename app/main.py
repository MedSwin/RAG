from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_database, get_database, close_database
from app.core.state import initialize_services, get_model_manager, get_model_download_service, cleanup_services
from app.api.v1.router import api_router
from app.api.cds import router as cds_router
from app.api.auth import AuthMiddleware
from app.services.dataset import HuggingFaceDatasetService
from app.services.storage import StorageService
from app.medswin.readiness import cloud_health_status
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

initialize_services()


def _disabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _strict_benchmark_runtime() -> bool:
    """Return True for the publication matrix API subprocesses."""
    return bool(str(os.getenv("BENCHMARK_ORG_ID") or "").strip()) and bool(settings.CLOUD_MODE)


def _active_generation_identity() -> tuple[str, str]:
    """Return the actual generation backend/model selected by LLMClient."""
    backend = settings.resolved_generation_backend()
    if backend == "medswin_local":
        return backend, os.getenv("MEDSWIN_LLM_MODEL") or settings.MEDSWIN_LLM_MODEL
    if backend == "bedrock":
        return backend, os.getenv("BEDROCK_CHAT_MODEL") or settings.BEDROCK_CHAT_MODEL
    if backend == "foundry":
        return backend, os.getenv("FOUNDRY_MODEL") or settings.FOUNDRY_MODEL or settings.CLOUD_MODEL
    return backend, "default"


def _validate_cloud_configuration() -> None:
    """Fail before serving traffic when cloud mode is structurally unusable.

    This validates configuration only and deliberately does not spend provider
    quota. ``system-eval`` performs the stronger live profile probes.
    """
    if not settings.CLOUD_MODE:
        return
    if settings.cloud_provider() == "bedrock":
        missing = []
        if not str(settings.AWS_BEDROCK_REGION or "").strip():
            missing.append("AWS_BEDROCK_REGION")
        if not str(settings.AWS_BEDROCK_RERANK_REGION or "").strip():
            missing.append("AWS_BEDROCK_RERANK_REGION")
        if not str(settings.BEDROCK_EMBED_MODEL or "").strip():
            missing.append("BEDROCK_EMBED_MODEL")
        if not str(settings.BEDROCK_RERANK_MODEL or "").strip():
            missing.append("BEDROCK_RERANK_MODEL")
        if settings.BEDROCK_EMBED_DIMENSION not in {256, 512, 1024, 1536}:
            raise RuntimeError("BEDROCK_EMBED_DIMENSION must be 256, 512, 1024, or 1536")
        backend, model = _active_generation_identity()
        if backend == "bedrock" and not str(model or "").strip():
            missing.append("BEDROCK_CHAT_MODEL")
        if missing:
            raise RuntimeError(
                "Cloud mode is missing required configuration: " + ", ".join(sorted(set(missing)))
            )
        return
    missing = []
    if not str(settings.AZURE_AI_FOUNDRY_API_KEY or "").strip():
        missing.append("AZURE_AI_FOUNDRY_API_KEY")
    try:
        embedding_url = settings.cloud_embedding_url()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cloud embedding configuration is invalid: {exc}") from exc
    try:
        reranker_url = settings.cloud_reranker_url()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cloud reranker configuration is invalid: {exc}") from exc
    if not str(embedding_url or "").strip():
        missing.append("CLOUD_EMBEDDING_URI or AZURE_AI_FOUNDRY_ENDPOINT")
    if not str(reranker_url or "").strip():
        missing.append("CLOUD_RERANKER_URI or AZURE_AI_FOUNDRY_ENDPOINT")
    backend, model = _active_generation_identity()
    if backend == "foundry":
        if not str(settings.AZURE_AI_FOUNDRY_ENDPOINT or "").strip():
            missing.append("AZURE_AI_FOUNDRY_ENDPOINT")
        if not str(model or "").strip():
            missing.append("FOUNDRY_MODEL")
    if missing:
        raise RuntimeError("Cloud mode is missing required configuration: " + ", ".join(sorted(set(missing))))


async def _cleanup_endpoint_runtimes() -> None:
    """Close endpoint singletons that own HTTP clients and ANN mappings."""
    try:
        from app.api.v1.endpoints.pipelines import cleanup_pipeline_orchestrators

        await cleanup_pipeline_orchestrators()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pipeline client cleanup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting RAG application...")

    try:
        await init_database()
        logger.info("Database initialized")
    except Exception as e:
        if _strict_benchmark_runtime():
            logger.error("Strict benchmark startup requires MongoDB; refusing to start: %s", e)
            raise
        logger.warning("Database connection failed: %s", e)
        logger.warning("Application will start but database features may not be available")

    if settings.CLOUD_MODE:
        _validate_cloud_configuration()
        logger.info("Cloud mode configuration validated; skipping local HF embedding/reranker load")
    else:
        try:
            logger.info("Checking and downloading models...")
            model_download_service = get_model_download_service()
            download_results = await model_download_service.download_all_models()
            for model_name, result in download_results.items():
                if result["success"]:
                    logger.info("Model %s: %s", model_name, result["message"])
                else:
                    logger.warning("Model %s: %s", model_name, result.get("error", "Unknown error"))
        except Exception as e:
            logger.error("Failed to download models: %s", e)

        try:
            model_manager = get_model_manager()
            await model_manager.load_embedding_model()
            await model_manager.load_reranker_model()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error("Failed to load models: %s", e)
            logger.warning("Continuing without models - some features may not be available")

    async def preload_datasets():
        """Preload dashboard dataset information without blocking normal startup."""
        try:
            logger.info("Preloading dataset information from Hugging Face...")
            hf_service = HuggingFaceDatasetService()
            try:
                stats = await asyncio.wait_for(
                    hf_service.get_total_statistics(use_cache=False),
                    timeout=300.0,
                )
                logger.info(
                    "Dataset preloading completed: %s datasets, %s total rows",
                    stats['total_datasets'],
                    stats['total_rows'],
                )
            except asyncio.TimeoutError:
                logger.warning("Dataset preloading timed out - will load on demand")
            except Exception as e:
                logger.warning("Dataset preloading failed: %s - will load on demand", e)
        except Exception as e:
            logger.warning("Error setting up dataset preloading: %s - will load on demand", e)

    if _disabled("DISABLE_DATASET_PRELOAD"):
        logger.info("Dataset dashboard preload disabled for this process")
    else:
        asyncio.create_task(preload_datasets())

    try:
        yield
    finally:
        logger.info("Shutting down RAG application...")
        await _cleanup_endpoint_runtimes()
        try:
            from app.api.v1.endpoints.storage import cleanup_storage_service

            cleanup_storage_service()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Storage worker cleanup failed: %s", exc)
        try:
            from app.api.v1.endpoints.preprocessing import cleanup_preprocessing_service

            cleanup_preprocessing_service()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Preprocessing worker cleanup failed: %s", exc)
        cleanup_services()
        await close_database()


app = FastAPI(
    title="MedSwin Clinical Decision Support",
    description="Evidence-gated multi-agent clinical RAG with sufficiency audit",
    version=settings.VERSION,
    lifespan=lifespan
)

def cors_allow_credentials(origins: list[str]) -> bool:
    """Allow cookies only for an explicit origin list."""
    return bool(origins) and "*" not in origins


app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=cors_allow_credentials(list(settings.ALLOWED_ORIGINS)),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")
app.include_router(cds_router)

_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
_WEB_PUBLIC = Path(__file__).resolve().parent.parent / "web" / "public"
if _WEB_DIST.is_dir():
    app.mount("/app", StaticFiles(directory=str(_WEB_DIST), html=True), name="clinician")
elif _WEB_PUBLIC.is_dir():
    app.mount("/app", StaticFiles(directory=str(_WEB_PUBLIC), html=True), name="clinician")


@app.get("/")
async def root():
    """Root endpoint - clinician CDSS UI when present, else dashboard."""
    from fastapi.responses import RedirectResponse
    if _WEB_DIST.is_dir() or _WEB_PUBLIC.is_dir():
        return RedirectResponse(url="/app/")
    return RedirectResponse(url="/api/v1/dashboard/")


@app.get("/health")
async def health_check():
    """Health check including real database and generation identities."""
    try:
        db = get_database()
        await db.command("ping")

        generation_backend, generation_model = _active_generation_identity()
        if settings.CLOUD_MODE:
            storage = StorageService()
            refresh = storage.get_embedding_refresh_status()
            try:
                stats = await storage.get_storage_stats()
                index_ready = bool(stats.get("index_exists")) and bool(stats.get("index_provenance_valid", True))
            except Exception:  # noqa: BLE001
                stats = {}
                index_ready = False
            status = cloud_health_status(
                index_exists=bool(stats.get("index_exists")),
                provenance_valid=bool(stats.get("index_provenance_valid")),
            )
            body = {
                "status": status,
                "contract_version": "2.0",
                "supported_pipelines": ["naive", "two-stage", "full"],
                "system_eval_mode": _disabled("SYSTEM_EVAL_MODE"),
                "cloud_mode": True,
                "generation_backend": generation_backend,
                "generation_model": generation_model,
                "resolved_local_model_revision": os.getenv("MEDSWIN_MODEL_REVISION") or settings.MEDSWIN_MODEL_REVISION,
                "llm_model": generation_model,
                "embedding_model": settings.active_embedding_model(),
                "reranker_model": settings.active_reranker_model(),
                "active_embedding_space": settings.active_embedding_space(),
                "active_embedding_dimension": settings.active_embedding_dimension(),
                "active_index_ready": index_ready,
                "index_provenance_error": stats.get("index_provenance_error"),
                "embedding_refresh": refresh,
                "dataset_preload_disabled": _disabled("DISABLE_DATASET_PRELOAD"),
                "database": "connected",
            }
            if status == "degraded":
                return JSONResponse(status_code=503, content=jsonable_encoder(body))
            return body

        model_manager = get_model_manager()
        embedding_loaded = model_manager.embedding_model is not None
        reranker_loaded = model_manager.reranker_model is not None
        storage = StorageService()
        try:
            stats = await storage.get_storage_stats()
            index_ready = bool(stats.get("index_exists")) and bool(stats.get("index_provenance_valid", True))
            status = cloud_health_status(
                index_exists=bool(stats.get("index_exists")),
                provenance_valid=bool(stats.get("index_provenance_valid", True)),
            )
        except Exception:  # noqa: BLE001
            stats = {}
            index_ready = False
            status = "healthy"
        body = {
            "status": status,
            "contract_version": "2.0",
            "supported_pipelines": ["naive", "two-stage", "full"],
            "system_eval_mode": _disabled("SYSTEM_EVAL_MODE"),
            "cloud_mode": False,
            "generation_backend": generation_backend,
            "generation_model": generation_model,
            "resolved_local_model_revision": os.getenv("MEDSWIN_MODEL_REVISION") or settings.MEDSWIN_MODEL_REVISION,
            "llm_model": generation_model,
            "embedding_model": "loaded" if embedding_loaded else "not_loaded",
            "reranker_model": "loaded" if reranker_loaded else "not_loaded",
            "active_embedding_space": settings.active_embedding_space(),
            "active_embedding_dimension": settings.active_embedding_dimension(),
            "active_index_ready": index_ready,
            "index_provenance_error": stats.get("index_provenance_error"),
            "dataset_preload_disabled": _disabled("DISABLE_DATASET_PRELOAD"),
            "database": "connected",
        }
        if status == "degraded":
            return JSONResponse(status_code=503, content=jsonable_encoder(body))
        return body
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
