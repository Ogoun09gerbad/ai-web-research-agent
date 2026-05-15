"""Utility helpers for validation, formatting, and logging."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from app.config import AppConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure a consistent application-wide logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_valid_url(url: str) -> bool:
    """Validate that a string looks like a usable HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def normalize_urls(urls: str | Iterable[str]) -> list[str]:
    """
    Normalize one or multiple URLs.

    Accepts:
    - a newline/comma-separated string
    - any iterable of strings

    Returns a de-duplicated list of valid URLs in input order.
    """
    if isinstance(urls, str):
        parts = re.split(r"[\n,]+", urls)
        raw_urls = [p.strip() for p in parts if p.strip()]
    else:
        raw_urls = [str(u).strip() for u in urls if str(u).strip()]

    seen: set[str] = set()
    cleaned: list[str] = []
    for url in raw_urls:
        if is_valid_url(url) and url not in seen:
            cleaned.append(url)
            seen.add(url)
    return cleaned


def clean_text(text: str) -> str:
    """Collapse repeated whitespace and remove noisy blank lines."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Trim long strings for UI display."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def safe_session_name(session_id: str) -> str:
    """Convert a session identifier into a safe filesystem / collection name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id.strip())
    return safe[:80] or "default_session"


def hash_text(text: str) -> str:
    """Generate a stable short hash for IDs and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_cache_key(url: str) -> str:
    """Build a compact cache key for a URL."""
    return hash_text(url)


def app_title() -> str:
    """Central project title."""
    return "AI Web Research Agent"


def default_user_agent() -> str:
    """A descriptive user agent for polite scraping."""
    return f"{app_title()}/1.0 (+FastAPI +Streamlit)"


def load_app_config() -> AppConfig:
    """Convenience wrapper to load the app config."""
    from app.config import get_settings

    return get_settings()
