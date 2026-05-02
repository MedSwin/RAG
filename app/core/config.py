from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings for MedSwin."""
    
    # Application settings
    APP_NAME: str = "MedSwin"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server settings
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8100  # Changed from 8000 to avoid conflict with supervisor
    HOST: str = "0.0.0.0"  # Legacy alias
    PORT: int = 8100  # Legacy alias
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "medswin"
    MONGODB_DATABASE: str = "medswin"  # Legacy alias
    
    # Model endpoints (OpenAI-compatible)
    SUPERVISOR_URL: str = "http://localhost:8000/v1/chat/completions"
    AGENT1_URL: str = "http://localhost:8001/v1/chat/completions"
    AGENT2_URL: str = "http://localhost:8002/v1/chat/completions"
    AGENT3_URL: str = "http://localhost:8003/v1/chat/completions"
    RERANKER_URL: str = "http://localhost:8004/rerank"
    EMBEDDING_URL: str = "http://localhost:8005/embeddings"
    
    # Service timeouts
    LLM_TIMEOUT_S: int = 60
    RERANK_TIMEOUT_S: int = 30
    EMBED_TIMEOUT_S: int = 30
    
    # Legacy model settings (for backward compatibility)
    EMBEDDING_MODEL_PATH: str = "./models/MedEmbed-large-v0.1"
    RERANKER_MODEL_PATH: str = "./models/bge-reranker-v2-m3"
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Storage settings
    HNSW_INDEX_PATH: str = "./data/hnsw_index.bin"
    HNSW_MAPPING_PATH: str = "./data/hnsw_mapping.json"
    FAISS_INDEX_PATH: str = "./data/faiss_index.bin"
    FAISS_MAPPING_PATH: str = "./data/faiss_mapping.json"
    TREE_INDEX_PATH: str = "./data/tree_index.npy"
    TREE_MAPPING_PATH: str = "./data/tree_mapping.json"
    DATA_DIR: str = "./data"
    
    # Retrieval settings
    DEFAULT_TOP_K: int = 5
    CANDIDATE_K: int = 80
    CANDIDATE_K_PRIME: int = 120
    MAX_RETRIEVE_LOOPS: int = 3
    TOKEN_BUDGET_B: int = 1800
    ENABLE_BM25: bool = True
    DEFAULT_INDEX_TYPE: str = "hnsw"
    INDEX_STRATEGY_MODE: str = "dynamic"
    
    # Legacy retrieval settings (for backward compatibility)
    MAX_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    FINAL_TOP_K: int = 3
    
    # Evidence sufficiency policy
    SUFF_T_CPG: int = 2
    SUFF_T_EMR: int = 2
    SUFF_T_INCLUSION: float = 0.55
    SUFF_T_MEAN_CONF: float = 0.60
    SUFF_FACET_THRESHOLD: float = 0.70
    SUFF_CRITICAL_FACET_THRESHOLD: float = 0.78
    SUFF_LCB_MARGIN: float = 0.08
    SUFF_MAX_ENTROPY: float = 0.88
    SUFF_MAX_CONTRADICTIONS: int = 0
    SUFF_MIN_MARGINAL_UTILITY: float = 0.0002
    DEFAULT_CLINICAL_SCOPE: str = "clinician_cds"
    
    # Fusion score weights (must sum to 1.0)
    W_RERANK: float = 0.45
    W_DENSE: float = 0.25
    W_LEX: float = 0.10
    W_RECENCY: float = 0.07
    W_SECTION: float = 0.08
    W_SOURCE: float = 0.05
    W_EBM: float = 0.12
    W_NOISE: float = 0.20
    FUSION_LOGIT_CLIP: float = 8.0
    RECENCY_DECAY_DAYS: float = 730.0
    SECTION_RECOMMENDATION_SCORE: float = 1.0
    SECTION_BACKGROUND_SCORE: float = 0.30
    SECTION_DEFAULT_SCORE: float = 0.65
    SOURCE_CPG_SCORE: float = 0.95
    SOURCE_EMR_SCORE: float = 0.85
    SOURCE_LIT_SCORE: float = 0.75
    EBM_CPG_WEIGHT: float = 0.95
    EBM_SR_WEIGHT: float = 0.90
    EBM_RCT_WEIGHT: float = 0.86
    EBM_OBS_WEIGHT: float = 0.62
    EBM_CASE_WEIGHT: float = 0.38
    EBM_EMR_WEIGHT: float = 0.70
    EBM_SAFETY_WEIGHT: float = 0.88
    SAFETY_REWARD_WEIGHT: float = 0.35
    REDUNDANCY_PENALTY_WEIGHT: float = 0.22
    CONTRADICTION_PENALTY_WEIGHT: float = 0.40
    NOISE_PENALTY_WEIGHT: float = 0.30

    # MMR configuration
    MMR_LAMBDA: float = 0.75
    MMR_MAX_EVIDENCE_CHUNKS: int = 10
    
    # Enterprise features
    ENABLE_AUTH: bool = False
    ENABLE_RBAC: bool = False
    ENABLE_OTEL: bool = False
    LOG_REDACT_PHI: bool = True
    TRACE_REDACT_BY_DEFAULT: bool = True
    TRACE_INCLUDE_POLICY_DETAILS: bool = True
    
    # Chunking settings
    TARGET_CHUNK_SIZE: int = 400
    BATCH_SIZE: int = 64
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".txt", ".pdf"]
    
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
    
    def validate_fusion_weights(self) -> bool:
        """Validate that base fusion weights are within a stable range."""
        total = (
            self.W_RERANK + self.W_DENSE + self.W_LEX +
            self.W_RECENCY + self.W_SECTION + self.W_SOURCE
        )
        return 0.10 <= total <= 2.0

    def validate_enterprise_policy(self) -> bool:
        """Validate policy thresholds that gate clinician-facing answers."""
        thresholds = [
            self.SUFF_T_INCLUSION,
            self.SUFF_T_MEAN_CONF,
            self.SUFF_FACET_THRESHOLD,
            self.SUFF_CRITICAL_FACET_THRESHOLD,
            self.SUFF_MAX_ENTROPY,
        ]
        if any(value < 0.0 or value > 1.0 for value in thresholds):
            return False
        if self.TOKEN_BUDGET_B <= 0 or self.MAX_RETRIEVE_LOOPS <= 0:
            return False
        if self.FUSION_LOGIT_CLIP <= 0:
            return False
        return True

# Create settings instance
settings = Settings()

# Validate fusion weights on startup
if not settings.validate_fusion_weights():
    import warnings
    warnings.warn(
        f"Fusion base weights are outside the stable range (sum={settings.W_RERANK + settings.W_DENSE + settings.W_LEX + settings.W_RECENCY + settings.W_SECTION + settings.W_SOURCE}). "
        "This may cause unexpected scoring behavior."
    )

if not settings.validate_enterprise_policy():
    raise ValueError("Invalid MedSwin enterprise policy configuration")

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
