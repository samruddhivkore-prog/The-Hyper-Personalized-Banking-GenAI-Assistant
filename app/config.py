from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./banking_assistant.db"

    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "stub"  # stub | groq | gemini | ollama
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    ollama_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"

    sentry_dsn: str = ""
    log_level: str = "INFO"

    fairness_parity_threshold: float = 0.10

    segmentation_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
