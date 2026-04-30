from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = "qwen3.5-omni-plus"
    model_provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""

    step_timeout: int = 30
    step_max_retries: int = 3
    retry_base_delay: float = 1.0

    env: str = "prod"
    database_url: str = ""
    checkpoint_database_url: str = ""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24
    faiss_index_path: str = "data/faiss.index"
    memory_meta_path: str = "data/memory_meta.json"
    memory_top_k: int = 5
    memory_token_limit: int = 2000
    memory_rebuild: bool = False
    embedding_model: str = "tongyi-embedding-vision-plus"
    embedding_api_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )

    tools_dir: str = str(Path(__file__).parent.parent / "tools")

    app_title: str = "AI Agent Plugin Platform"
    app_version: str = "1.0.0"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    app_description: str = "AI Agent plugin platform built with FastAPI and LangGraph."

    @field_validator("jwt_secret_key")
    @classmethod
    def jwt_secret_key_must_be_configured(cls, value: str) -> str:
        value = value.strip()
        if not value or value == "change-this-secret-key-in-production":
            raise ValueError("JWT_SECRET_KEY must be configured to a real secret value.")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, (list, tuple)):
                        return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("cors_origins must be a list or comma-separated string.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
