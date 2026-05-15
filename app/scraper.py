"""Web scraping utilities for fetching and cleaning webpage content."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import AppConfig, get_settings
from app.utils import build_cache_key, clean_text, default_user_agent, is_valid_url

logger = logging.getLogger(__name__)


class ScrapingError(Exception):
    """Raised when a webpage cannot be fetched or parsed."""


@dataclass
class ScrapedPage:
    """A cleaned webpage ready for RAG indexing."""

    url: str
    title: str
    text: str
    html: str
    fetched_at: float
    status_code: int | None = None


class WebScraper:
    """Fetch webpages, clean HTML, and cache responses locally."""

    def __init__(self, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_file = Path(self.settings.cache_dir) / "web_cache.json"
        self.cache: dict[str, dict[str, Any]] = self._load_cache()

        self.session = requests.Session()
        retry = Retry(
            total=self.settings.max_retries,
            connect=self.settings.max_retries,
            read=self.settings.max_retries,
            status=self.settings.max_retries,
            backoff_factor=self.settings.retry_backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": default_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def scrape_urls(self, urls: list[str]) -> tuple[list[ScrapedPage], list[str]]:
        """Scrape multiple URLs and continue even if some fail."""
        pages: list[ScrapedPage] = []
        errors: list[str] = []

        for url in urls:
            try:
                pages.append(self.scrape_url(url))
            except Exception as exc:
                msg = f"{url}: {exc}"
                logger.exception("Failed to scrape %s", url)
                errors.append(msg)

        return pages, errors

    def scrape_url(self, url: str) -> ScrapedPage:
        """Scrape a single URL and return cleaned content."""
        url = url.strip()
        if not is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        cached = self._get_cached_page(url)
        if cached:
            logger.info("Cache hit for %s", url)
            return ScrapedPage(**cached)

        logger.info("Fetching %s", url)
        try:
            response = self.session.get(url, timeout=self.settings.request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ScrapingError(f"Failed to fetch {url}: {exc}") from exc

        html = response.text
        title, text = self.extract_clean_text(html, base_url=url)

        page = ScrapedPage(
            url=url,
            title=title,
            text=text,
            html=html,
            fetched_at=time.time(),
            status_code=response.status_code,
        )
        self._set_cache(url, page)
        return page

    def extract_clean_text(self, html: str, base_url: str = "") -> tuple[str, str]:
        """Remove noisy tags and extract readable text from raw HTML."""
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form", "svg", "canvas", "button", "input", "select", "img"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = clean_text(soup.title.string)

        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator="\n", strip=True)
        text = clean_text(text)

        if not text:
            raise ScrapingError(f"No readable content found for {base_url or 'page'}")

        return title or "Untitled Page", text

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load the on-disk cache if it exists."""
        if not self.cache_file.exists():
            return {}
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Cache file was unreadable; starting fresh.")
            return {}

    def _save_cache(self) -> None:
        """Persist cache entries to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cached_page(self, url: str) -> dict[str, Any] | None:
        """Return a cached page if it is still fresh."""
        key = build_cache_key(url)
        item = self.cache.get(key)
        if not item:
            return None

        cached_at = float(item.get("fetched_at", 0))
        ttl_seconds = self.settings.cache_ttl_hours * 3600
        if time.time() - cached_at > ttl_seconds:
            return None

        return item

    def _set_cache(self, url: str, page: ScrapedPage) -> None:
        """Store a page in the cache."""
        key = build_cache_key(url)
        self.cache[key] = {
            "url": page.url,
            "title": page.title,
            "text": page.text,
            "html": page.html,
            "fetched_at": page.fetched_at,
            "status_code": page.status_code,
        }
        self._save_cache()
