"""Streamlit frontend for the AI Web Research Agent.

BUG FIXED — DuplicateWidgetID crash in api_url():
  The original `api_url(path)` function called `st.sidebar.text_input(...)` on
  every invocation.  Because call_api() is called multiple times per script run
  (ingest, ask, reset), Streamlit raised a DuplicateWidgetID error: you cannot
  render the same widget label more than once per run.

  Fix: The "Backend API URL" text_input is defined ONCE in the sidebar block
  with an explicit `key="api_base_url"`.  Streamlit persists the value across
  reruns via st.session_state automatically.  call_api() reads the value from
  st.session_state instead of re-rendering a widget.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# Ensure the repository root is on PYTHONPATH so `import app.*` works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.utils import normalize_urls, truncate_text

settings = get_settings()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Web Research Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Web Research Agent")
st.caption("Scrape websites, index them with RAG, and ask questions in a modern chat UI.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_urls" not in st.session_state:
    st.session_state.indexed_urls = []

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = settings.api_base_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_api(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the FastAPI backend and return JSON data.

    BUG FIX: reads the base URL from st.session_state (set once in sidebar)
    instead of calling st.sidebar.text_input() on every invocation.
    """
    base = st.session_state.api_base_url
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    response = requests.request(method, url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Sidebar  — rendered ONCE per script run
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Project Controls")
    st.write(f"**Session ID:** `{st.session_state.session_id[:12]}...`")

    # BUG FIX: text_input defined here ONCE with an explicit key so Streamlit
    # stores the value in session_state["api_base_url"] automatically.
    st.text_input(
        "Backend API URL",
        key="api_base_url",
        help="URL of the running FastAPI backend.",
    )

    urls_text = st.text_area(
        "Webpage URLs",
        placeholder="https://example.com\nhttps://example.org/article",
        height=140,
    )

    ingest_clicked = st.button("Index webpages", use_container_width=True)
    reset_clicked = st.button("Reset session", use_container_width=True)

    st.markdown("---")
    st.subheader("Current Indexed URLs")
    if st.session_state.indexed_urls:
        for url in st.session_state.indexed_urls:
            st.write(f"- {truncate_text(url, 70)}")
    else:
        st.caption("No webpages indexed yet.")


# ---------------------------------------------------------------------------
# Reset handler
# ---------------------------------------------------------------------------

if reset_clicked:
    try:
        with st.spinner("Resetting session..."):
            call_api("DELETE", f"/reset/{st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.indexed_urls = []
        st.success("Session reset.")
    except Exception as exc:
        st.error(f"Reset failed: {exc}")


# ---------------------------------------------------------------------------
# Ingest handler
# ---------------------------------------------------------------------------

if ingest_clicked:
    urls = normalize_urls(urls_text)
    if not urls:
        st.error("Please enter one or more valid URLs.")
    else:
        try:
            with st.spinner("Scraping and indexing webpages..."):
                result = call_api(
                    "POST",
                    "/ingest",
                    {
                        "session_id": st.session_state.session_id,
                        "urls": urls,
                    },
                )
            st.session_state.indexed_urls = urls
            st.success(
                f"Indexed {result['pages_scraped']} page(s) and "
                f"{result['chunks_indexed']} chunks."
            )
            if result.get("errors"):
                with st.expander("Scraping warnings"):
                    for err in result["errors"]:
                        st.write(f"- {err}")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.markdown("### Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(f"**{src.get('title', 'Untitled')}**")
                    st.write(src.get("url", ""))
                    st.write(f"Score: {src.get('score', 0):.4f}")
                    st.markdown("---")

question = st.chat_input("Ask a question about the indexed webpages...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_api(
                    "POST",
                    "/ask",
                    {
                        "session_id": st.session_state.session_id,
                        "question": question,
                    },
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.write(f"**{src.get('title', 'Untitled')}**")
                            st.write(src.get("url", ""))
                            st.write(f"Score: {src.get('score', 0):.4f}")
                            st.markdown("---")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except Exception as exc:
        error_message = f"Sorry, something went wrong: {exc}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
