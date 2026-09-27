"""Unit tests for fin_ai.core.report_pipeline."""

from __future__ import annotations

import pytest

from fin_ai.core import report_pipeline as rp_module
from fin_ai.core.exceptions import DocumentProcessingError
from fin_ai.core.rag import RetrievedChunk


def test_format_metrics_handles_error_dict():
    text = rp_module._format_metrics({"error": "Unable to retrieve market data."})
    assert "No market metrics available" in text
    assert "Unable to retrieve market data." in text


def test_format_metrics_reports_insufficient_as_na_not_fabricated():
    metrics = {
        "symbol": "NVDA",
        "returns_pct": {"1w": 1.0, "1m": None, "3m": None, "6m": None, "1y": None},
        "volatility_annualized_pct": 40.0,
        "ma_20": 120.5,
        "ma_50": None,
        "max_drawdown_pct": -15.0,
        "latest_price": 130.0,
    }
    text = rp_module._format_metrics(metrics)
    assert "1Y Return: N/A" in text
    assert "insufficient historical data" in text
    assert "50-Day Moving Average: N/A" in text
    # Real values are passed through exactly, not recalculated.
    assert "40.00%" in text
    assert "120.50" in text


def test_generate_research_report_passes_evidence_and_metrics_to_llm(monkeypatch):
    chunk = RetrievedChunk(text="Cloud revenue grew sharply.", source="10k.pdf", page=12, section=None, score=0.8)
    monkeypatch.setattr(rp_module, "search", lambda *a, **k: [chunk])

    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        captured["max_tokens"] = kwargs.get("max_tokens")
        return "# Executive Summary\n\nThe company performed well."

    monkeypatch.setattr(rp_module, "chat", fake_chat)

    metrics = {
        "symbol": "MSFT",
        "returns_pct": {"1w": None, "1m": None, "3m": None, "6m": None, "1y": 24.7},
        "volatility_annualized_pct": 22.0,
        "ma_20": 400.0,
        "ma_50": 390.0,
        "max_drawdown_pct": -10.0,
        "latest_price": 410.0,
    }

    report = rp_module.generate_research_report("Microsoft", doc_ids=["doc1"], metrics=metrics)

    user_msg = captured["messages"][1]["content"]
    assert "10k.pdf" in user_msg
    assert "24.70%" in user_msg  # exact metric value passed through, not re-derived
    assert "Executive Summary" in report
    assert "10k.pdf" in report  # sources footer includes the evidence source
    assert "not financial advice" in report.lower()


def test_generate_research_report_with_no_documents_still_works(monkeypatch):
    monkeypatch.setattr(rp_module, "chat", lambda *a, **k: "Report body")
    report = rp_module.generate_research_report("Some Company", doc_ids=[], metrics=None)
    assert "Report body" in report
    assert "No document evidence was used" in report


def test_generate_research_report_wraps_retrieval_failure(monkeypatch):
    """Regression test, mirroring test_qa.py: a raw exception from search()
    must surface as a specific DocumentProcessingError, not an opaque,
    unlogged failure the page can't explain to the user.
    """

    def boom(*a, **k):
        raise RuntimeError("Failed to load embedding model 'BAAI/bge-small-en-v1.5'.")

    monkeypatch.setattr(rp_module, "search", boom)
    with pytest.raises(DocumentProcessingError, match="Failed to load embedding model"):
        rp_module.generate_research_report("Some Company", doc_ids=["doc1"], metrics=None)
