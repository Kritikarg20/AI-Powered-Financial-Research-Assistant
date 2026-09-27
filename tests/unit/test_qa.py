"""Unit tests for fin_ai.core.qa — grounded document Q&A."""

from __future__ import annotations

import pytest

from fin_ai.core import qa as qa_module
from fin_ai.core.exceptions import DocumentProcessingError
from fin_ai.core.rag import RetrievedChunk


def test_answer_question_no_docs_selected():
    result = qa_module.answer_question("What was revenue?", doc_ids=[])
    assert result.insufficient_evidence is True
    assert "No documents are selected" in result.answer


def test_answer_question_empty_question():
    result = qa_module.answer_question("   ", doc_ids=["doc1"])
    assert result.insufficient_evidence is True


def test_answer_question_no_chunks_found(monkeypatch):
    monkeypatch.setattr(qa_module, "search", lambda *a, **k: [])
    result = qa_module.answer_question("Anything?", doc_ids=["doc1"])
    assert result.insufficient_evidence is True
    assert "do not contain enough information" in result.answer


def test_answer_question_grounds_on_retrieved_chunks(monkeypatch):
    chunk = RetrievedChunk(text="Revenue grew 12% year over year.", source="annual.pdf", page=5, section=None, score=0.9)
    monkeypatch.setattr(qa_module, "search", lambda *a, **k: [chunk])

    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return "Revenue grew 12% year over year, according to the filing."

    monkeypatch.setattr(qa_module, "chat", fake_chat)

    result = qa_module.answer_question("What happened to revenue?", doc_ids=["doc1"])
    assert result.sources == [chunk]
    assert "12%" in result.answer
    # The context sent to the LLM must include the retrieved evidence and its source.
    user_msg = captured["messages"][1]["content"]
    assert "annual.pdf" in user_msg
    assert "page 5" in user_msg
    assert "Revenue grew 12%" in user_msg


def test_answer_question_wraps_retrieval_failure(monkeypatch):
    """Regression test: a raw exception from search() (e.g. the local
    embedder failing to load) must not reach the page as an opaque,
    unlogged error. It should surface as a specific DocumentProcessingError
    so the UI can show something more useful than "Something went wrong".
    """

    def boom(*a, **k):
        raise RuntimeError("Failed to load embedding model 'BAAI/bge-small-en-v1.5'.")

    monkeypatch.setattr(qa_module, "search", boom)
    with pytest.raises(DocumentProcessingError, match="Failed to load embedding model"):
        qa_module.answer_question("What happened to revenue?", doc_ids=["doc1"])
