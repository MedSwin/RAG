import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelDownloadService:
    """Service for downloading models from Hugging Face."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Model configurations
        self.models = {
            "embedding": {
                "repo_id": "MedAI-COS30018/MedEmbed-large-v0.1",
                "local_path": settings.EMBEDDING_MODEL_PATH,
                "type": "embedding"
            },
            "reranker": {
                "repo_id": "BAAI/bge-reranker-v2-m3",
                "local_path": settings.RERANKER_MODEL_PATH,
                "type": "reranker"
            }
        }
    
    async def download_model(self, model_name: str) -> Dict[str, Any]:
        """Download a model from Hugging Face."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models[model_name]
        
        try:
            logger.info(f"Downloading model {model_name} from {model_config['repo_id']}")
            
            # Run download in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._download_model_sync,
                model_config
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "local_path": model_config["local_path"],
                "message": f"Successfully downloaded {model_name}"
            }
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e)
            }
    
    def _download_model_sync(self, model_config: Dict[str, Any]) -> None:
        """Synchronous model download."""
        try:
            local_path = Path(model_config["local_path"])
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download the entire model repository
            snapshot_download(
                repo_id=model_config["repo_id"],
                local_dir=str(local_path),
                token=self.hf_token,
                resume_download=True
            )
            
            logger.info(f"Successfully downloaded model to {local_path}")
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    async def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        if model_name not in self.models:
            return False
        
        model_config = self.models[model_name]
        local_path = Path(model_config["local_path"])
        
        # Check if model directory exists and has required files
        if not local_path.exists():
            return False
        
        # Check for required files based on model type
        if model_config["type"] == "embedding":
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        elif model_config["type"] == "reranker":
            required_files = ["config.json", "pytorch_model.bin"]
        else:
            required_files = ["config.json"]
        
        for file_name in required_files:
            if not (local_path / file_name).exists():
                return False
        
        return True
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models[model_name]
        exists = await self.check_model_exists(model_name)
        
        return {
            "name": model_name,
            "repo_id": model_config["repo_id"],
            "local_path": model_config["local_path"],
            "type": model_config["type"],
            "exists": exists,
            "size_gb": await self._get_model_size(model_name) if exists else 0
        }
    
    async def _get_model_size(self, model_name: str) -> float:
        """Get the size of a downloaded model in GB."""
        try:
            model_config = self.models[model_name]
            local_path = Path(model_config["local_path"])
            
            if not local_path.exists():
                return 0
            
            total_size = 0
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return round(total_size / (1024**3), 2)  # Convert to GB
            
        except Exception as e:
            logger.error(f"Error getting model size: {e}")
            return 0
    
    async def download_all_models(self) -> Dict[str, Any]:
        """Download all required models."""
        results = {}
        
        for model_name in self.models.keys():
            exists = await self.check_model_exists(model_name)
            
            if not exists:
                logger.info(f"Model {model_name} not found locally, downloading...")
                result = await self.download_model(model_name)
                results[model_name] = result
            else:
                logger.info(f"Model {model_name} already exists locally")
                results[model_name] = {
                    "success": True,
                    "model_name": model_name,
                    "message": "Model already exists locally"
                }
        
        return results
    
    async def get_all_models_info(self) -> Dict[str, Any]:
        """Get information about all models."""
        models_info = {}
        
        for model_name in self.models.keys():
            try:
                models_info[model_name] = await self.get_model_info(model_name)
            except Exception as e:
                logger.error(f"Error getting info for model {model_name}: {e}")
                models_info[model_name] = {
                    "name": model_name,
                    "error": str(e)
                }
        
        return models_info
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
