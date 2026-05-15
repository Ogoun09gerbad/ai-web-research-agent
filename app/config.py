"""Application configuration and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Central configuration for the AI Web Research Agent."""

    # Project paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    chroma_dir: Path = Path(__file__).resolve().parent.parent / "data" / "chroma"
    cache_dir: Path = Path(__file__).resolve().parent.parent / "data" / "cache"
    logs_dir: Path = Path(__file__).resolve().parent.parent / "data" / "logs"

    # LLM / embeddings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Scraper
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "20"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff: float = float(os.getenv("RETRY_BACKOFF", "0.5"))
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "24"))

    # API / UI
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Return a cached configuration instance."""
    settings = AppConfig()
    ensure_directories(settings)
    return settings


def ensure_directories(settings: AppConfig | None = None) -> None:
    """Create required directories if they do not exist."""
    cfg = settings or AppConfig()
    for path in (cfg.data_dir, cfg.chroma_dir, cfg.cache_dir, cfg.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
