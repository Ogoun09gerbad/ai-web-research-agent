"""Embedding manager using sentence-transformers through LangChain.

BUG FIXED:
  `get_embedding_manager()` used `AppConfig(**{**settings.__dict__, ...})` to
  override a single field.  This is fragile: `__dict__` on a frozen dataclass
  also contains internal dunder keys, and it bypasses dataclass field
  validation entirely.  The correct idiom is `dataclasses.replace()`, which
  creates a new frozen instance with only the specified fields changed.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import AppConfig, get_settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Loads and exposes a sentence-transformers embedding model.

    The model is lazy-loaded on first use so startup is fast and the heavy
    sentence-transformers download only happens when actually needed.
    """

    def __init__(self, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()
        self._embeddings: HuggingFaceEmbeddings | None = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model on first access."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self.settings.embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return self.embeddings.embed_documents(texts)


@lru_cache(maxsize=4)
def get_embedding_manager(model_name: str | None = None) -> EmbeddingManager:
    """Return a cached EmbeddingManager, optionally overriding the model name.

    BUG FIX: Original code used `AppConfig(**{**settings.__dict__, ...})`.
    Using `dataclasses.replace()` is the correct, validated way to produce a
    modified copy of a frozen dataclass instance.
    """
    settings = get_settings()
    if model_name:
        settings = dataclasses.replace(settings, embedding_model=model_name)
    return EmbeddingManager(settings=settings)
