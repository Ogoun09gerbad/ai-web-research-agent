"""ChromaDB vector store wrapper for document storage and retrieval.

BUG FIXED: This file was a completely empty stub (`class VectorStoreManager: pass`).
Every other module called .add_documents(), .similarity_search(),
.similarity_search_with_score(), and .clear() — all of which were missing,
causing AttributeError crashes at runtime the moment any agent was used.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a per-session Chroma vector store for document storage and retrieval.

    Design notes:
    - The Chroma client is lazy-initialised on first use to avoid loading the
      embedding model before the FastAPI startup sequence completes.
    - Each agent session receives its own collection name and persist directory
      so that memory is fully isolated between users.
    - `clear()` deletes the collection and resets the internal reference so the
      next call to `.store` recreates a fresh, empty collection automatically.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_manager: EmbeddingManager,
        collection_name: str = "ai_web_research",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self._store: Chroma | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def store(self) -> Chroma:
        """Lazy-initialise the Chroma store on first access."""
        if self._store is None:
            logger.info(
                "Initialising Chroma store | collection=%s | path=%s",
                self.collection_name,
                self.persist_directory,
            )
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_manager.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store documents; return the assigned IDs.

        Chroma requires every document to have a unique string ID.
        We generate UUIDs here so callers do not have to.
        """
        if not documents:
            logger.warning("add_documents called with an empty list — nothing stored.")
            return []

        ids = [uuid.uuid4().hex for _ in documents]
        self.store.add_documents(documents, ids=ids)
        logger.info("Stored %d document chunks in collection '%s'.", len(documents), self.collection_name)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the k most similar documents for a query string."""
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return (Document, relevance_score) pairs.

        Uses `similarity_search_with_relevance_scores` which returns scores in
        [0, 1] (higher = more relevant), unlike the raw distance variant which
        returns lower-is-better L2 distances.  This keeps downstream score
        comparisons intuitive.
        """
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def clear(self) -> None:
        """Delete all documents from this collection and reset the store reference.

        After clear(), the next call to .store will create a fresh, empty
        Chroma collection automatically — no manual re-initialisation needed.
        """
        try:
            self.store.delete_collection()
            logger.info("Deleted Chroma collection '%s'.", self.collection_name)
        except Exception as exc:
            logger.warning(
                "Could not delete collection '%s': %s — resetting reference anyway.",
                self.collection_name,
                exc,
            )
        finally:
            self._store = None

    def document_count(self) -> int:
        """Return the number of indexed document chunks (useful for health checks)."""
        try:
            return self.store._collection.count()  # noqa: SLF001
        except Exception:
            return 0
