"""
Shared helpers for the Financial Research AI dashboard pages.

Each page calls :func:`render_sidebar_status` to show a small, consistent
status panel (Mistral configured?, embedding model, document count) and
imports config constants from here rather than duplicating them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
SRC_DIR = PARENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from fin_ai.config.fin_ai import (  # noqa: E402
    EMBEDDING_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    YAHOO_SERVICE_OFFLINE,
)
from fin_ai.core.rag import list_documents  # noqa: E402

logger = logging.getLogger(__name__)

APP_TITLE = "Financial Research AI"


def render_sidebar_status() -> None:
    """Render a compact, read-only status panel in the sidebar.

    This app has no per-provider shopping list in the sidebar (Mistral is
    the only LLM, embeddings are local) — configuration lives in the
    Settings page and the .env file. This panel just orients the user.
    """
    with st.sidebar:
        st.caption("Status")
        st.write("Mistral API Key:", "✅ Configured" if MISTRAL_API_KEY else "❌ Not configured")
        st.write("Model:", MISTRAL_MODEL)
        st.write("Embedding model:", EMBEDDING_MODEL)
        st.write("Market data:", "Offline sample data" if YAHOO_SERVICE_OFFLINE else "Live (yfinance)")
        try:
            docs = list_documents()
            st.write("Documents indexed:", len(docs))
        except Exception:
            logger.exception("Failed to read indexed document count for sidebar status")
            st.write("Documents indexed:", "N/A")
        st.divider()
        st.caption("Financial Research AI — educational project, not financial advice.")
