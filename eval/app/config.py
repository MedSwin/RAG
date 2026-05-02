from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Benchmark harness configuration."""

    medswin_base_url: str = Field(default="http://localhost:8100", alias="MEDSWIN_BASE_URL")
    benchmark_org_id: str = Field(default="bench-org", alias="BENCHMARK_ORG_ID")
    benchmark_user_id: str = Field(default="bench-user", alias="BENCHMARK_USER_ID")
    request_timeout_s: float = Field(default=120.0, alias="REQUEST_TIMEOUT_S")
    default_token_budget: int = Field(default=4096, alias="DEFAULT_TOKEN_BUDGET")
    run_store_dir: str = Field(default="/app/audits", alias="RUN_STORE_DIR")
    max_cases_default: int = Field(default=25, alias="MAX_CASES_DEFAULT")

    class Config:
        env_file = ".env"
        populate_by_name = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
