"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Stores runtime configuration read from environment variables.
class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DBMIND_", extra="ignore")

    database_url: str
    database_schema: str | None = Field(default="public")
    schema_context_path: str = Field(default="./data/schema_context.json")
    business_context_path: str = Field(default="./data/internal_docs")
    chroma_path: str = Field(default="./data/chroma")
    chroma_collection_prefix: str = Field(default="dbmind")
    sql_dialect: str = Field(default="postgres")
    min_execution_score: int = Field(default=90, ge=0, le=100)
    max_corrections: int = Field(default=2, ge=0, le=10)
    default_limit: int = Field(default=100, ge=1, le=10_000)
    llm_model: str = Field(default="google_genai:gemini-3.6-flash")
    llm_provider: str | None = Field(default=None)
    llm_temperature: float | None = Field(default=None, ge=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)


# Returns cached application settings for reuse across the app.
@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
