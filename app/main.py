from fastapi import FastAPI, HTTPException
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
from app.api.auth import AuthMiddleware
from app.services.dataset import HuggingFaceDatasetService
from app.services.storage import StorageService
from app.services.adapters.llm import _generation_backend
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
    backend = _generation_backend()
    if backend == "medswin_local":
        return backend, os.getenv("MEDSWIN_LLM_MODEL") or settings.MEDSWIN_LLM_MODEL
    if backend == "foundry":
        return backend, os.getenv("FOUNDRY_MODEL") or settings.FOUNDRY_MODEL or settings.CLOUD_MODEL
    return backend, "default"


def _validate_cloud_configuration() -> None:
    """Fail before serving traffic when cloud mode is structurally unusable.

    This validates configuration only and deliberately does not spend provider
    quota. The publication `full-eval` warmup performs the stronger live probes.
    """
    if not settings.CLOUD_MODE:
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
        if not str(settings.FOUNDRY_ENDPOINT or settings.AZURE_AI_FOUNDRY_ENDPOINT or "").strip():
            missing.append("FOUNDRY_ENDPOINT or AZURE_AI_FOUNDRY_ENDPOINT")
        if not str(model or "").strip():
            missing.append("FOUNDRY_MODEL")
    if missing:
        raise RuntimeError("Cloud mode is missing required configuration: " + ", ".join(sorted(set(missing))))


async def _cleanup_endpoint_runtimes() -> None:
    """Close endpoint singletons that own HTTP clients and ANN mappings."""
    try:
        from app.api.v1.endpoints.naive import cleanup_naive_orchestrator

        await cleanup_naive_orchestrator()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Naive RAG client cleanup failed: %s", exc)

    try:
        from app.api.v1.endpoints import medswin as medswin_endpoint

        orchestrator = getattr(medswin_endpoint, "_orchestrator", None)
        if orchestrator is not None:
            await orchestrator.close()
            medswin_endpoint._orchestrator = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("MedSwin client cleanup failed: %s", exc)


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

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")

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
            refresh = StorageService().get_embedding_refresh_status()
            return {
                "status": "healthy",
                "cloud_mode": True,
                "generation_backend": generation_backend,
                "generation_model": generation_model,
                "llm_model": generation_model,
                "embedding_model": settings.CLOUD_EMBEDDING,
                "reranker_model": settings.CLOUD_RERANKER,
                "active_embedding_space": settings.active_embedding_space(),
                "active_embedding_dimension": settings.active_embedding_dimension(),
                "active_index_ready": refresh.get("ready", False),
                "embedding_refresh": refresh,
                "dataset_preload_disabled": _disabled("DISABLE_DATASET_PRELOAD"),
                "database": "connected",
            }

        model_manager = get_model_manager()
        embedding_loaded = model_manager.embedding_model is not None
        reranker_loaded = model_manager.reranker_model is not None
        return {
            "status": "healthy",
            "cloud_mode": False,
            "generation_backend": generation_backend,
            "generation_model": generation_model,
            "llm_model": generation_model,
            "embedding_model": "loaded" if embedding_loaded else "not_loaded",
            "reranker_model": "loaded" if reranker_loaded else "not_loaded",
            "dataset_preload_disabled": _disabled("DISABLE_DATASET_PRELOAD"),
            "database": "connected",
        }
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