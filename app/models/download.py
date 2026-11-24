import os
import logging
from typing import Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

logger = logging.getLogger(__name__)


def _has_weights(local_path: Path) -> bool:
    """Accept a single-file or sharded weight directory."""
    if (local_path / "model.safetensors").exists() or (local_path / "pytorch_model.bin").exists():
        return True
    return any(local_path.glob("model*.safetensors")) or any(local_path.glob("pytorch_model*.bin"))


def _has_tokenizer(local_path: Path) -> bool:
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "sentencepiece.bpe.model")
    return any((local_path / name).exists() for name in names)


class ModelDownloadService:
    """Service for downloading models from Hugging Face."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Model configurations
        self.models = {
            "embedding": {
                "repo_id": "abhinand/MedEmbed-large-v0.1",
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
            from huggingface_hub import snapshot_download

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
        
        if model_config["type"] == "embedding":
            return _has_weights(local_path) and _has_tokenizer(local_path) and (local_path / "config.json").exists()
        if model_config["type"] == "reranker":
            return _has_weights(local_path) and (local_path / "config.json").exists()
        return (local_path / "config.json").exists()
    
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

