"""Unit tests for fin_ai.core.market — graceful error handling wrapper."""

from __future__ import annotations

import pandas as pd

from fin_ai.core import market as market_module
from fin_ai.core.exceptions import MarketDataNotFoundError, MarketDataServiceError


class _FakeService:
    def __init__(self, mode: str):
        self.mode = mode

    def get_stock_data(self, symbol, start, end):
        if self.mode == "not_found":
            raise MarketDataNotFoundError(symbol, "stock_data")
        if self.mode == "service_error":
            raise MarketDataServiceError("network down")
        if self.mode == "unexpected":
            raise ValueError("boom")
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.date_range("2026-01-01", periods=3))


def test_get_price_history_empty_symbol():
    df, error = market_module.get_price_history("   ")
    assert df is None
    assert "ticker" in error.lower()


def test_get_price_history_not_found(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("not_found"))
    df, error = market_module.get_price_history("FAKE")
    assert df is None
    assert "Unable to retrieve market data" in error
    assert "FAKE" in error


def test_get_price_history_service_error_no_traceback_leak(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("service_error"))
    df, error = market_module.get_price_history("AAPL")
    assert df is None
    assert "network down" not in error.lower()  # internal exception detail not leaked verbatim
    assert "try again later" in error.lower()


def test_get_price_history_unexpected_exception_handled(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("unexpected"))
    df, error = market_module.get_price_history("AAPL")
    assert df is None
    assert "boom" not in error  # raw exception text never shown to the user
    assert error


def test_get_price_history_success(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("ok"))
    df, error = market_module.get_price_history("AAPL")
    assert error is None
    assert not df.empty


def test_get_metrics_propagates_error_dict(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("not_found"))
    metrics = market_module.get_metrics("FAKE")
    assert metrics["error"]
    assert metrics["symbol"] == "FAKE"


def test_get_metrics_success_uses_python_calculation(monkeypatch):
    monkeypatch.setattr(market_module, "get_service", lambda: _FakeService("ok"))
    metrics = market_module.get_metrics("AAPL")
    assert metrics["symbol"] == "AAPL"
    assert "returns_pct" in metrics
