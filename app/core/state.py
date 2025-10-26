"""
Global application state management.
This module provides a centralized way to access shared resources
without creating circular imports.
"""

from typing import Optional
from app.services.model_manager import ModelManager
from app.services.model_download_service import ModelDownloadService

# Global instances
model_manager: Optional[ModelManager] = None
model_download_service: Optional[ModelDownloadService] = None

def initialize_services():
    """Initialize global services."""
    global model_manager, model_download_service
    
    if model_manager is None:
        model_manager = ModelManager()
    
    if model_download_service is None:
        model_download_service = ModelDownloadService()

def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    if model_manager is None:
        raise RuntimeError("Model manager not initialized. Call initialize_services() first.")
    return model_manager

def get_model_download_service() -> ModelDownloadService:
    """Get the global model download service instance."""
    if model_download_service is None:
        raise RuntimeError("Model download service not initialized. Call initialize_services() first.")
    return model_download_service

def cleanup_services():
    """Cleanup global services."""
    global model_manager, model_download_service
    
    if model_manager is not None:
        model_manager.cleanup()
        model_manager = None
    
    if model_download_service is not None:
        model_download_service.cleanup()
        model_download_service = None
