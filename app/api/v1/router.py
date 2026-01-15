from fastapi import APIRouter
from app.api.v1.endpoints import embedding, preprocessing, retrieval, storage, dashboard, medswin

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(embedding.router, prefix="/embedding", tags=["embedding"])
api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["preprocessing"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(storage.router, prefix="/storage", tags=["storage"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(medswin.router, prefix="/medswin", tags=["medswin"])
