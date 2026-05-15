"""RAG pipeline: chunking, retrieval, and answer generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AppConfig, get_settings
from app.embeddings import EmbeddingManager
from app.scraper import ScrapedPage, WebScraper
from app.utils import truncate_text
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured answer plus supporting sources."""

    answer: str
    sources: list[dict[str, str | float]]


class RAGPipeline:
    """High-level orchestration for indexing and question answering."""

    def __init__(
        self,
        settings: AppConfig | None = None,
        scraper: WebScraper | None = None,
        embedding_manager: EmbeddingManager | None = None,
        vector_store: VectorStoreManager | None = None,
        llm: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scraper = scraper or WebScraper(self.settings)
        self.embedding_manager = embedding_manager or EmbeddingManager(self.settings)
        self.vector_store = vector_store
        self.llm = llm

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def set_vector_store(self, vector_store: VectorStoreManager) -> None:
        """Attach a vector store after instantiation if needed."""
        self.vector_store = vector_store

    def split_page(self, page: ScrapedPage) -> list[Document]:
        """Convert a page into LangChain Documents suitable for Chroma."""
        chunks = self.splitter.split_text(page.text)
        documents: list[Document] = []

        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": page.url,
                        "title": page.title,
                        "chunk": idx,
                        "text_preview": truncate_text(chunk, 250),
                    },
                )
            )
        return documents

    def ingest_urls(self, urls: list[str]) -> dict[str, object]:
        """Scrape URLs, chunk text, and store the documents."""
        if self.vector_store is None:
            raise RuntimeError("Vector store is not configured.")

        pages, errors = self.scraper.scrape_urls(urls)
        documents: list[Document] = []

        for page in pages:
            documents.extend(self.split_page(page))

        ids = self.vector_store.add_documents(documents)
        return {
            "pages_scraped": len(pages),
            "chunks_indexed": len(documents),
            "document_ids": ids,
            "errors": errors,
        }

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """Retrieve the most relevant chunks for a question."""
        if self.vector_store is None:
            raise RuntimeError("Vector store is not configured.")

        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search(question, k=top_k)

    def retrieve_with_scores(self, question: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Retrieve chunks with their similarity scores."""
        if self.vector_store is None:
            raise RuntimeError("Vector store is not configured.")

        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search_with_score(question, k=top_k)

    def format_context(self, docs: Iterable[Document]) -> str:
        """Build a readable context block from retrieved chunks."""
        blocks: list[str] = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            title = meta.get("title", "Untitled")
            blocks.append(
                f"[Source {i}] {title}\nURL: {source}\nChunk: {doc.page_content.strip()}"
            )
        return "\n\n".join(blocks)

    def create_prompt(self) -> ChatPromptTemplate:
        """Create the QA prompt template used by the chatbot."""
        system = (
            "You are an intelligent AI web research assistant. "
            "Answer only from the provided context whenever possible. "
            "If the context does not contain the answer, say so clearly. "
            "Be concise, accurate, and cite the source titles when helpful."
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Give a clear, helpful answer grounded in the context.",
                ),
            ]
        )

    def build_sources(self, docs: list[Document], scores: list[float] | None = None) -> list[dict[str, str | float]]:
        """Turn retrieved documents into a compact source list."""
        sources: list[dict[str, str | float]] = []
        for idx, doc in enumerate(docs):
            meta = doc.metadata or {}
            score = scores[idx] if scores and idx < len(scores) else 0.0
            sources.append(
                {
                    "title": str(meta.get("title", "Untitled")),
                    "url": str(meta.get("source", "")),
                    "chunk": float(meta.get("chunk", 0)),
                    "score": float(score),
                }
            )
        return sources

    def format_retrieved_snippets(self, docs: list[Document]) -> str:
        """Create a compact snippet view for logging or debugging."""
        parts = []
        for doc in docs:
            parts.append(truncate_text(doc.page_content, 180))
        return " | ".join(parts)
