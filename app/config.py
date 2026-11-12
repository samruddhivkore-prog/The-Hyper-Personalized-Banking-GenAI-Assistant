from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./banking_assistant.db"

    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "stub"  # stub | anthropic | groq | gemini | ollama
    llm_model: str = ""  # overrides the provider's default model name if set
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    ollama_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"

    sentry_dsn: str = ""
    log_level: str = "INFO"

    fairness_parity_threshold: float = 0.10

    segmentation_k: int = 5

    admin_password: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
