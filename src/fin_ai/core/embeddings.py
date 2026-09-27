"""
Local embeddings — fastembed (ONNX runtime), no external server required.

This replaces the previous Ollama-based embeddings implementation.
``fastembed`` runs small ONNX embedding models locally in-process — no
PyTorch/CUDA, no ``ollama serve``, no local server, no network calls at
query time. The model file is downloaded once (cached under
``~/.cache/fastembed``) the first time it is used.

Usage::

    from fin_ai.core.embeddings import get_embedder

    embedder = get_embedder()
    vectors = embedder.embed_documents(["chunk one", "chunk two"])
    query_vector = embedder.embed_query("what was revenue?")
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Iterable

from fin_ai.config.fin_ai import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()


class LocalEmbedder:
    """Wraps a local ``fastembed`` model behind a small, stable API.

    Exposes ``embed_documents(list[str]) -> list[list[float]]`` and
    ``embed_query(str) -> list[float]`` so it is a drop-in fit for the
    FAISS-backed RAG store in :mod:`fin_ai.core.rag`.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with _load_lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "The 'fastembed' package is required for local embeddings. "
                    "Install project dependencies with `pip install -r requirements.txt`."
                ) from exc
            try:
                logger.info("Loading local embedding model '%s'...", self.model_name)
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}'. "
                    "Check your internet connection (needed the first time to "
                    f"download the model) and that the model name is correct. "
                    f"Underlying error: {exc}"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vec = self.embed_query("dimension probe")
            self._dimension = len(vec)
        return self._dimension

    @staticmethod
    def _normalize(vector) -> list[float]:
        import numpy as np

        arr = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        model = self._ensure_loaded()
        vectors = list(model.embed(texts))
        return [self._normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        vectors = list(model.query_embed([text]))
        return self._normalize(vectors[0])

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


@lru_cache(maxsize=4)
def _cached_embedder(model_name: str) -> LocalEmbedder:
    return LocalEmbedder(model_name=model_name)


def get_embedder(model_name: str | None = None) -> LocalEmbedder:
    """Return a (process-cached) :class:`LocalEmbedder` for *model_name*.

    Falls back to the configured default (``EMBEDDING_MODEL``) when no
    model name is supplied. The underlying model itself is lazily loaded
    on first use.
    """
    return _cached_embedder(model_name or EMBEDDING_MODEL)
