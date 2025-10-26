from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "Medical RAG System"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "medical_rag_db"
    
    # Model settings
    EMBEDDING_MODEL_PATH: str = "./models/MedEmbed-large-v0.1"
    RERANKER_MODEL_PATH: str = "./models/bge-reranker-v2-m3"
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Storage settings
    HNSW_INDEX_PATH: str = "./data/hnsw_index.bin"
    HNSW_MAPPING_PATH: str = "./data/hnsw_mapping.json"
    DATA_DIR: str = "./data"
    
    # Retrieval settings
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    FINAL_TOP_K: int = 3
    
    # Chunking settings
    TARGET_CHUNK_SIZE: int = 400
    BATCH_SIZE: int = 64
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".txt"]
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Hugging Face settings
    HF_TOKEN: Optional[str] = None
    
    # AWS settings (for EC2 deployment)
    AWS_REGION: str = "us-east-1"
    EC2_INSTANCE_TYPE: str = "g4dn.xlarge"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        Path(settings.DATA_DIR),
        Path(settings.EMBEDDING_MODEL_PATH).parent,
        Path(settings.RERANKER_MODEL_PATH).parent,
        Path(settings.LOG_FILE).parent,
        Path("./models"),
        Path("./data"),
        Path("./logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()
