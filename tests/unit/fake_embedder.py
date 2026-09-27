"""A tiny, deterministic, dependency-free fake embedder for tests.

Real embeddings (fastembed) require downloading a model from the internet
the first time they run, which is not available in every test
environment. This fake embedder implements the same interface
(``embed_documents`` / ``embed_query``) using a simple deterministic
character-hashing scheme, so RAG/FAISS mechanics (chunking, indexing,
retrieval, multi-document search, metadata) can be tested without any
network access or heavy ML dependency.
"""

from __future__ import annotations

import re

import numpy as np

DIM = 64


def _vectorize(text: str) -> list[float]:
    vec = np.zeros(DIM, dtype="float32")
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word in words:
        idx = hash(word) % DIM
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class FakeEmbedder:
    model_name = "fake-test-embedder"

    def embed_documents(self, texts):
        return [_vectorize(t) for t in texts]

    def embed_query(self, text):
        return _vectorize(text)

    def __call__(self, text):
        return self.embed_query(text)
