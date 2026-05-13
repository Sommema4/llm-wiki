from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# Always resolve .env from the project root regardless of working directory
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    metacentrum_base_url: str = "https://llm.ai.e-infra.cz/v1"
    # OpenRouter model names
    default_model: str = "google/gemini-2.0-flash-001"
    chat_model: str = "anthropic/claude-sonnet-4-5"
    # MetaCentrum model names (override via METACENTRUM_DEFAULT_MODEL / METACENTRUM_CHAT_MODEL in .env)
    metacentrum_default_model: str = "deepseek-v4-pro-thinking"
    metacentrum_chat_model: str = "deepseek-v4-pro-thinking"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    upload_dir: str = "./uploads"
    database_url: str = "sqlite:///./llm_wiki.db"
    max_text_chars: int = 500000

    model_config = {"env_file": str(_ENV_FILE)}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
