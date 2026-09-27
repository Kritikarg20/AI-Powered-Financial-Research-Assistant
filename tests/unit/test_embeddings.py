"""Unit tests for fin_ai.core.embeddings.

Mocks the fastembed backend so these tests run without downloading any
model or requiring network access, while still exercising the real
LocalEmbedder wrapper logic (normalization, batching, dimension property).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fin_ai.core.embeddings import LocalEmbedder


class _FakeTextEmbedding:
    """Minimal fake of fastembed.TextEmbedding: deterministic 8-dim vectors."""

    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        for t in texts:
            yield np.array([float(len(t) % 7 + 1)] * 8)

    def query_embed(self, texts):
        for t in texts:
            yield np.array([float(len(t) % 7 + 1)] * 8)


@pytest.fixture(autouse=True)
def fake_fastembed(monkeypatch):
    fake_module = types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    yield


def test_embed_query_returns_normalized_vector():
    embedder = LocalEmbedder(model_name="fake-model")
    vector = embedder.embed_query("hello")
    assert len(vector) == 8
    norm = float(np.linalg.norm(vector))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_documents_returns_one_vector_per_text():
    embedder = LocalEmbedder(model_name="fake-model")
    vectors = embedder.embed_documents(["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


def test_embed_documents_empty_list_returns_empty():
    embedder = LocalEmbedder(model_name="fake-model")
    assert embedder.embed_documents([]) == []


def test_dimension_property():
    embedder = LocalEmbedder(model_name="fake-model")
    assert embedder.dimension == 8


def test_model_loaded_lazily_only_once():
    embedder = LocalEmbedder(model_name="fake-model")
    assert embedder._model is None
    embedder.embed_query("x")
    loaded_model = embedder._model
    assert loaded_model is not None
    embedder.embed_query("y")
    assert embedder._model is loaded_model  # not reloaded


def test_get_embedder_raises_friendly_error_without_fastembed(monkeypatch):
    monkeypatch.delitem(sys.modules, "fastembed", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no module named fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embedder = LocalEmbedder(model_name="fake-model")
    with pytest.raises(RuntimeError, match="fastembed"):
        embedder.embed_query("hi")
