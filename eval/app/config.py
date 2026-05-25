from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Benchmark harness configuration."""

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")

    medswin_base_url: str = Field(default="http://localhost:8100", alias="MEDSWIN_BASE_URL")
    benchmark_org_id: str = Field(default="bench-org", alias="BENCHMARK_ORG_ID")
    benchmark_user_id: str = Field(default="bench-user", alias="BENCHMARK_USER_ID")
    request_timeout_s: float = Field(default=120.0, alias="REQUEST_TIMEOUT_S")
    default_token_budget: int = Field(default=4096, alias="DEFAULT_TOKEN_BUDGET")
    # Root Cause vs Logic: the UI path previously defaulted to /app/audits, which
    # breaks when the benchmark container or host mount makes /app read-only.
    # Logic: keep audit artifacts in a writable cache location by default and let
    # deployments override it explicitly when they want a different persistence mount.
    run_store_dir: str = Field(default="/tmp/medswin-audits", alias="RUN_STORE_DIR")
    max_cases_default: int = Field(default=25, alias="MAX_CASES_DEFAULT")
    benchmark_random_seed: int = Field(default=1337, alias="BENCHMARK_RANDOM_SEED")
    benchmark_corpus_sample_size: int = Field(default=5000, alias="BENCHMARK_CORPUS_SAMPLE_SIZE")
    benchmark_max_topics: int = Field(default=30, alias="BENCHMARK_MAX_TOPICS")
    benchmark_min_gold_corpus_recall: float = Field(default=0.95, alias="BENCHMARK_MIN_GOLD_CORPUS_RECALL")
    benchmark_min_gold_index_recall: float = Field(default=0.95, alias="BENCHMARK_MIN_GOLD_INDEX_RECALL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
