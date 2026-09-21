from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from app.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    app_name: str = "KnowledgeHub"

    environment: Literal[
        "local",
        "test",
        "production",
    ] = "local"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_embedding_model: str = (
        "qwen3-embedding:8b"
    )

    embedding_dimensions: int = Field(
        default=1024,
        gt=0,
    )

    # PostgreSQL / pgvector
    postgres_url: SecretStr

    vector_schema: str = "public"
    vector_table_name: str = "rag_documents"

    # Retrieval
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Chunking
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=10_000,
    )

    chunk_overlap: int = Field(
        default=120,
        ge=0,
    )

    # Uploads
    document_root: Path = Path(
        "./app/data/documents"
    )

    max_upload_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KNOWLEDGEHUB_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()

    except ValidationError as exc:
        raise ConfigurationError(
            "KnowledgeHub configuration validation failed."
        ) from exc