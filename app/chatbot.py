"""Chatbot service that combines scraping, RAG, and memory.

BUGS FIXED:
1. CRITICAL — Broken chain assignment in `ask()`:
       chain = prompt | self.llm | self.pipeline.create_prompt().output_parser
   `ChatPromptTemplate` has no `.output_parser` attribute.  This line raised
   AttributeError immediately, making the try/except block below it *unreachable*.
   The fix is to remove this dead line entirely and keep only the correct chain
   built with StrOutputParser inside the try/except.

2. MINOR — `VectorStoreManager` was a stub (fixed in vector_store.py), so
   `self.vector_store.clear()` in `reset()` would also have crashed.

3. REPLACED OpenAI with Groq (free) to avoid quota/billing issues.
   Get a free API key at: https://console.groq.com/keys

4. FIXED — Model `llama3-8b-8192` has been decommissioned by Groq.
   Replaced with `llama-3.1-8b-instant` which is the current supported equivalent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import AppConfig, get_settings
from app.embeddings import EmbeddingManager
from app.memory import MemoryManager
from app.rag_pipeline import RAGPipeline, RetrievalResult
from app.utils import safe_session_name
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result returned after indexing URLs."""

    pages_scraped: int
    chunks_indexed: int
    document_ids: list[str]
    errors: list[str]


class AIWebResearchAgent:
    """One agent per session: its own memory and its own Chroma store."""

    def __init__(self, session_id: str, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_id = safe_session_name(session_id)
        self.memory_manager = MemoryManager()
        self.embedding_manager = EmbeddingManager(self.settings)

        collection_name = f"ai_web_research_{self.session_id}"
        persist_dir = self.settings.chroma_dir / self.session_id
        self.vector_store = VectorStoreManager(
            persist_directory=persist_dir,
            embedding_manager=self.embedding_manager,
            collection_name=collection_name,
        )

        self.llm = self._build_llm()
        self.pipeline = RAGPipeline(
            settings=self.settings,
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            llm=self.llm,
        )

    def _build_llm(self) -> ChatGroq:
        """Create the Groq chat model used for answering questions."""
        if not self.settings.openai_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Please configure your Groq API key in the .env file. "
                "Get a free key at: https://console.groq.com/keys"
            )

        # ✅ FIX: replaced decommissioned model 'llama3-8b-8192'
        #         with the current supported equivalent 'llama-3.1-8b-instant'
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=self.settings.openai_api_key,
            temperature=0.2,
        )

    def ingest_urls(self, urls: list[str]) -> IngestResult:
        """Scrape and index one or more URLs."""
        result = self.pipeline.ingest_urls(urls)
        return IngestResult(
            pages_scraped=int(result["pages_scraped"]),
            chunks_indexed=int(result["chunks_indexed"]),
            document_ids=list(result["document_ids"]),
            errors=list(result["errors"]),
        )

    def ask(self, question: str) -> RetrievalResult:
        """Answer a user question using retrieval-augmented generation."""
        docs_with_scores = self.pipeline.retrieve_with_scores(question, self.settings.top_k)
        docs = [doc for doc, _score in docs_with_scores]
        scores = [score for _doc, score in docs_with_scores]
        memory = self.memory_manager.get(self.session_id)
        history = memory.load_memory_variables({}).get("history", [])

        if not docs:
            answer = (
                "I could not find relevant indexed content yet. "
                "Please add one or more webpages first, then ask your question again."
            )
            self.memory_manager.save_turn(self.session_id, question, answer)
            return RetrievalResult(answer=answer, sources=[])

        context = self.pipeline.format_context(docs)
        prompt = self.pipeline.create_prompt()

        chain = prompt | self.llm | StrOutputParser()

        logger.info("Generating answer for session %s", self.session_id)
        answer = chain.invoke(
            {
                "history": history,
                "context": context,
                "question": question,
            }
        )

        self.memory_manager.save_turn(self.session_id, question, answer)
        sources = self.pipeline.build_sources(docs, scores)
        return RetrievalResult(answer=answer, sources=sources)

    def history(self) -> list[dict[str, str]]:
        """Return the session conversation history."""
        return self.memory_manager.history_as_dicts(self.session_id)

    def reset(self) -> None:
        """Clear the vector store and memory for this session."""
        self.memory_manager.clear(self.session_id)
        self.vector_store.clear()


class AgentRegistry:
    """Thread-safe cache of agents indexed by session ID."""

    def __init__(self, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()
        self._agents: dict[str, AIWebResearchAgent] = {}
        self._lock = Lock()

    def get_agent(self, session_id: str) -> AIWebResearchAgent:
        """Return a session-specific agent, creating it on first access."""
        session_key = safe_session_name(session_id)
        with self._lock:
            if session_key not in self._agents:
                self._agents[session_key] = AIWebResearchAgent(
                    session_id=session_key,
                    settings=self.settings,
                )
            return self._agents[session_key]

    def reset_agent(self, session_id: str) -> None:
        """Reset and remove an agent from the registry."""
        session_key = safe_session_name(session_id)
        with self._lock:
            agent = self._agents.get(session_key)
            if agent:
                agent.reset()
                del self._agents[session_key]