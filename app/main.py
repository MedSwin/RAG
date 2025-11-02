from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_database
from app.core.state import initialize_services, get_model_manager, get_model_download_service, cleanup_services
from app.api.v1.router import api_router
from app.services.hf_dataset_service import HuggingFaceDatasetService
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize global services
initialize_services()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting RAG application...")
    
    # Initialize database (with retries and graceful failure)
    try:
        await init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database connection failed (this is expected if security group not configured): {e}")
        logger.warning("Application will start but database features may not be available")
    
    # Download models if not present
    try:
        logger.info("Checking and downloading models...")
        model_download_service = get_model_download_service()
        download_results = await model_download_service.download_all_models()
        for model_name, result in download_results.items():
            if result["success"]:
                logger.info(f"Model {model_name}: {result['message']}")
            else:
                logger.warning(f"Model {model_name}: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        # Continue without failing startup
    
    # Load models
    try:
        model_manager = get_model_manager()
        await model_manager.load_embedding_model()
        await model_manager.load_reranker_model()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("Continuing without models - some features may not be available")
    
    # Preload dataset information in background (non-blocking)
    async def preload_datasets():
        """Preload dataset information in background."""
        try:
            logger.info("Preloading dataset information from Hugging Face...")
            hf_service = HuggingFaceDatasetService()
            
            # Preload all dataset info with timeout (use_cache=False to force fresh load)
            try:
                stats = await asyncio.wait_for(
                    hf_service.get_total_statistics(use_cache=False),
                    timeout=300.0  # 5 minute timeout
                )
                logger.info(f"Dataset preloading completed: {stats['total_datasets']} datasets, {stats['total_rows']} total rows")
                logger.info("Dataset statistics cached and ready for dashboard")
            except asyncio.TimeoutError:
                logger.warning("Dataset preloading timed out - will load on demand")
            except Exception as e:
                logger.warning(f"Dataset preloading failed: {e} - will load on demand")
        except Exception as e:
            logger.warning(f"Error setting up dataset preloading: {e} - will load on demand")
    
    # Start dataset preloading in background (don't wait for it)
    asyncio.create_task(preload_datasets())
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG application...")
    cleanup_services()

# Create FastAPI application
app = FastAPI(
    title="Medical RAG System",
    description="A comprehensive RAG system for medical document retrieval and question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint - redirects to dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/dashboard/")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are loaded
        model_manager = get_model_manager()
        embedding_loaded = model_manager.embedding_model is not None
        reranker_loaded = model_manager.reranker_model is not None
        
        return {
            "status": "healthy",
            "embedding_model": "loaded" if embedding_loaded else "not_loaded",
            "reranker_model": "loaded" if reranker_loaded else "not_loaded",
            "database": "connected"  # Add actual DB check if needed
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
