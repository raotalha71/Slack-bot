"""
Centralized configuration module.
Reads all settings from environment variables / .env file using Pydantic Settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Slack
    SLACK_BOT_TOKEN: str
    SLACK_SIGNING_SECRET: str
    SLACK_APP_TOKEN: str = ""  # Required for Socket Mode

    # LLM (Groq)
    GROQ_API_KEY: str
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Database
    DATABASE_URL: str = "sqlite:///./data/sessions.db"

    # Web Search Fallback (Tavily)
    TAVILY_API_KEY: str = ""

    # Application
    SEED_DATA_DIR: str = "seed_data"
    SIMILARITY_THRESHOLD: float = 0.45  # Min score for RAG match
    DEDUP_THRESHOLD: float = 0.85  # Max score before skipping save


@lru_cache
def get_settings() -> Settings:
    """
    Singleton settings instance.
    Cached so .env is only read once per process.
    """
    return Settings()
