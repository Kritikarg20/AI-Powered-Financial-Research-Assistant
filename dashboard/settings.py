"""Settings — read-only configuration summary."""

from dashboard.shared import render_sidebar_status
import streamlit as st

from fin_ai.config.fin_ai import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    RETRIEVAL_TOP_K,
    VECTOR_DB_DIR,
    YAHOO_DATA_DIR,
    YAHOO_SERVICE_OFFLINE,
)

render_sidebar_status()

st.title("⚙️ Settings")
st.caption("Configuration is environment-driven (.env). This page is read-only; edit .env and restart to change values.")

st.subheader("LLM Provider")
st.table(
    {
        "Setting": ["Provider", "Model", "API Key"],
        "Value": [
            DEFAULT_PROVIDER,
            MISTRAL_MODEL,
            "✅ Configured" if MISTRAL_API_KEY else "❌ Not configured — set MISTRAL_API_KEY in .env",
        ],
    }
)

st.subheader("Embeddings (local — no external server required)")
st.table(
    {
        "Setting": ["Embedding model", "Chunk size", "Chunk overlap", "Retrieval top-k"],
        "Value": [EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K],
    }
)

st.subheader("Market Data")
st.table(
    {
        "Setting": ["Mode", "Data directory"],
        "Value": [
            "Offline (bundled sample data)" if YAHOO_SERVICE_OFFLINE else "Live (yfinance)",
            YAHOO_DATA_DIR,
        ],
    }
)

st.subheader("Storage")
st.write(f"Vector index directory: `{VECTOR_DB_DIR}`")

st.divider()
st.info(
    "To change configuration: copy `.env.example` to `.env`, edit the values, "
    "and restart the Streamlit app. Never commit your `.env` file."
)
