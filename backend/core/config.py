from functools import lru_cache
from typing import Optional
import os

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


# Eagerly load .env so os.environ has all keys even without pydantic-settings
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


class Settings(BaseModel):
    # Core
    env: str = Field(default_factory=lambda: _env("ENV", "development"))
    app_host: str = Field(default_factory=lambda: _env("APP_HOST", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(_env("APP_PORT", "8000")))
    cors_origins: str = Field(default_factory=lambda: _env("CORS_ORIGINS", "*"))

    # LLM
    llm_provider: str = Field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))
    llm_model: str = Field(default_factory=lambda: _env("LLM_MODEL", "llama3-70b-8192"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: _env("GROQ_API_KEY"))

    # Embeddings (OpenAI or local)
    emb_provider: str = Field(default_factory=lambda: _env("EMB_PROVIDER", "openai"))
    emb_model: str = Field(default_factory=lambda: _env("EMB_MODEL", "text-embedding-3-small"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: _env("OPENAI_API_KEY"))

    # Storage & DB (Supabase)
    storage_backend: str = Field(default_factory=lambda: _env("STORAGE_BACKEND", "supabase"))
    supabase_url: Optional[str] = Field(default_factory=lambda: _env("SUPABASE_URL"))
    supabase_service_role_key: Optional[str] = Field(default_factory=lambda: _env("SUPABASE_SERVICE_ROLE_KEY"))
    supabase_storage_bucket: str = Field(default_factory=lambda: _env("SUPABASE_STORAGE_BUCKET", "tenders"))

    db_url: Optional[str] = Field(default_factory=lambda: _env("DB_URL"))
    db_sslmode: Optional[str] = Field(default_factory=lambda: _env("DB_SSLMODE"))

    # Security
    auth_secret: Optional[str] = Field(default_factory=lambda: _env("AUTH_SECRET"))
    retention_days: int = Field(default_factory=lambda: int(_env("RETENTION_DAYS", "30")))

    # RAG/Index
    data_root: str = Field(default_factory=lambda: _env("DATA_ROOT", "backend/data"))
    index_root: str = Field(default_factory=lambda: _env("INDEX_ROOT", "backend/data/index"))
    max_chunk_tokens: int = Field(default_factory=lambda: int(_env("MAX_CHUNK_TOKENS", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    top_k: int = Field(default_factory=lambda: int(_env("TOP_K", "5")))
    min_score: float = Field(default_factory=lambda: float(_env("MIN_SCORE", "0.3")))

    class Config:
        arbitrary_types_allowed = True

    @field_validator("llm_provider")
    def validate_llm_provider(cls, v):
        allowed = {"groq", "openai", "local"}
        if v not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}")
        return v

    @field_validator("storage_backend")
    def validate_storage_backend(cls, v):
        allowed = {"supabase", "local"}
        if v not in allowed:
            raise ValueError(f"STORAGE_BACKEND must be one of {allowed}")
        return v

    @field_validator("db_url")
    def require_db_url(cls, v):
        if not v:
            raise ValueError("DB_URL is required (use Supabase Postgres or local SQLite)")
        return v

    @field_validator("supabase_url", "supabase_service_role_key")
    def require_supabase_when_selected(cls, v, info):
        storage_backend = (info.data or {}).get("storage_backend") if hasattr(info, "data") else None
        if storage_backend == "supabase":
            if not v:
                # Identify which field is missing via the validator field name
                field_name = info.field_name if hasattr(info, "field_name") else "value"
                raise ValueError(f"{field_name.upper()} is required when STORAGE_BACKEND=supabase")
        return v

    @field_validator("groq_api_key")
    def require_groq_key_when_groq(cls, v, info):
        if (info.data or {}).get("llm_provider") == "groq" and not v:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return v

    @field_validator("openai_api_key")
    def require_openai_key_when_openai_embeddings(cls, v, info):
        if (info.data or {}).get("emb_provider") == "openai" and not v:
            raise ValueError("OPENAI_API_KEY is required when EMB_PROVIDER=openai")
        return v


@lru_cache()
def get_settings() -> Settings:
    # Construct from environment to avoid dependency on pydantic-settings
    return Settings()
