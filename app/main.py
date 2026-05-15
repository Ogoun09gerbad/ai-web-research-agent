"""FastAPI backend for the AI Web Research Agent.

BUG FIXED:
  `@app.on_event("startup")` is deprecated since FastAPI 0.93 and will be
  removed in a future release.  The correct pattern is a `lifespan` async
  context manager passed to the FastAPI constructor.  This also makes startup
  and shutdown logic testable and colocated.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chatbot import AgentRegistry
from app.config import get_settings
from app.utils import normalize_urls, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Handle startup and shutdown logic.

    BUG FIX: `@app.on_event("startup")` is deprecated in FastAPI >= 0.93.
    Using `lifespan` keeps startup/shutdown colocated, type-safe, and testable.
    """
    logger.info("Starting %s", app.title)
    yield
    logger.info("Shutting down %s", app.title)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Web Research Agent API",
    version="1.0.0",
    description="Scrape webpages, build a RAG index, and answer questions with memory.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = AgentRegistry(settings=settings)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Payload for indexing web pages."""

    urls: list[str] = Field(..., description="One or more valid HTTP/HTTPS URLs")
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class AskRequest(BaseModel):
    """Payload for a chat question."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class ResetRequest(BaseModel):
    """Payload for clearing a session."""

    session_id: str = Field(...)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest_webpages(payload: IngestRequest) -> dict[str, Any]:
    """Scrape and index one or more URLs."""
    urls = normalize_urls(payload.urls)
    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs were provided.")

    try:
        agent = registry.get_agent(payload.session_id)
        result = agent.ingest_urls(urls)
        return {
            "session_id": payload.session_id,
            "pages_scraped": result.pages_scraped,
            "chunks_indexed": result.chunks_indexed,
            "document_ids": result.document_ids,
            "errors": result.errors,
            "message": "Webpages successfully indexed.",
        }
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask_question(payload: AskRequest) -> dict[str, Any]:
    """Answer a question using the indexed webpages."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        agent = registry.get_agent(payload.session_id)
        result = agent.ask(payload.question.strip())
        return {
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": result.answer,
            "sources": result.sources,
            "history": agent.history(),
        }
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history/{session_id}")
def get_history(session_id: str) -> dict[str, Any]:
    """Return the current chat history for a session."""
    try:
        agent = registry.get_agent(session_id)
        return {"session_id": session_id, "history": agent.history()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/reset/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    """Clear memory and vector store for a session."""
    try:
        registry.reset_agent(session_id)
        return {"session_id": session_id, "message": "Session reset successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
