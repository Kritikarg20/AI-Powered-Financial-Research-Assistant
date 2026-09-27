"""
Application configuration — paths, model defaults, environment variables.

Everything the dashboard and ``fin_ai.core`` modules need lives here.
Environment variables (loaded from ``.env`` if present) override the
defaults below.

Import from this module directly::

    from fin_ai.config.fin_ai import VECTOR_DB_DIR, MISTRAL_MODEL, ...
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# __file__ = src/fin_ai/config/fin_ai.py
#   .parent               = src/fin_ai/config/
#   .parent.parent        = src/fin_ai/
#   .parent.parent.parent = src/
#   .parent.parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Load .env from the project root. Only secrets/overrides belong in .env;
# every other default lives in this file.
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_env_file, override=False)

VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "").strip() or str(PROJECT_ROOT / "vector_db")
PUBLISHED_RESEARCH_DIR = os.environ.get("PUBLISHED_RESEARCH_DIR", "").strip() or str(
    PROJECT_ROOT / "published_research"
)
DOCUMENTS_DIR = str(PROJECT_ROOT / "data" / "documents")

for _dir in (VECTOR_DB_DIR, PUBLISHED_RESEARCH_DIR, DOCUMENTS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Mistral AI / LiteLLM configuration (the LLM layer)
# ---------------------------------------------------------------------------

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest").strip()
MISTRAL_BASE_URL = os.environ.get("MISTRAL_BASE_URL", "").strip() or None
DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "mistral").strip().lower()

# LiteLLM model string. LiteLLM identifies the provider from a
# "<provider>/<model>" prefix; without it, `litellm.completion()` raises
# BadRequestError("LLM Provider NOT provided") before making any network
# call, regardless of whether MISTRAL_API_KEY is valid. MISTRAL_MODEL stays
# the bare, human-friendly name (used for display in Settings/sidebar);
# LITELLM_MODEL is the same model with the "mistral/" prefix LiteLLM
# requires and is what actually gets passed to litellm.completion().
LITELLM_MODEL = MISTRAL_MODEL if MISTRAL_MODEL.startswith("mistral/") else f"mistral/{MISTRAL_MODEL}"

# ---------------------------------------------------------------------------
# Embeddings (local, no external server required)
# ---------------------------------------------------------------------------

# A small, fast, ONNX-based local embedding model served by `fastembed`
# (no PyTorch/CUDA, no local server process). The model file downloads
# automatically on first use (cached under ~/.cache/fastembed) and then
# every embedding call runs fully offline in-process — no Ollama, no
# hosted embedding API required.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# RAG / chunking configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

YAHOO_SERVICE_OFFLINE = os.environ.get("YAHOO_SERVICE_OFFLINE", "true").strip().lower() == "true"
YAHOO_DATA_DIR = os.environ.get("YAHOO_DATA_DIR", "").strip() or str(PROJECT_ROOT / "data")

# ---------------------------------------------------------------------------
# Supported document types
# ---------------------------------------------------------------------------

SUPPORTED_UPLOAD_TYPES = ["pdf", "csv", "json", "html", "htm", "docx"]

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "VECTOR_DB_DIR",
    "PUBLISHED_RESEARCH_DIR",
    "DOCUMENTS_DIR",
    "MISTRAL_API_KEY",
    "MISTRAL_MODEL",
    "MISTRAL_BASE_URL",
    "DEFAULT_PROVIDER",
    "LITELLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
    "YAHOO_SERVICE_OFFLINE",
    "YAHOO_DATA_DIR",
    "SUPPORTED_UPLOAD_TYPES",
]
