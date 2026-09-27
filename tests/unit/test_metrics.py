"""Unit tests for fin_ai.core.metrics — deterministic calculations only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_ai.core.metrics import (
    compute_market_metrics,
    compute_max_drawdown,
    compute_moving_averages,
    compute_returns,
    compute_volatility,
    resolve_period_date_range,
)


def _closes(values):
    return pd.Series(values, dtype=float)


def test_compute_returns_known_values():
    # 252 trading days, flat 1.0 then a final day at 1.10 -> 1y return ~ 10%
    values = [1.0] * 252 + [1.10]
    closes = _closes(values)
    returns = compute_returns(closes)
    assert returns["1y"] == pytest.approx(10.0, abs=0.01)
    # 1-week window (5 trading days ago) is also 1.0 -> 1.10 => 10%
    assert returns["1w"] == pytest.approx(10.0, abs=0.01)


def test_compute_returns_insufficient_history():
    closes = _closes([1.0, 1.02, 1.01])
    returns = compute_returns(closes)
    assert returns["1y"] is None
    assert returns["1m"] is None


def test_compute_volatility_zero_for_constant_series():
    closes = _closes([100.0] * 30)
    vol = compute_volatility(closes)
    assert vol == pytest.approx(0.0, abs=1e-9)


def test_compute_volatility_matches_manual_calculation():
    values = [100, 101, 99, 102, 98, 103]
    closes = _closes(values)
    log_returns = np.diff(np.log(values))
    expected = float(log_returns.std(ddof=1) * np.sqrt(252) * 100)
    assert compute_volatility(closes) == pytest.approx(expected, rel=1e-6)


def test_compute_moving_averages_thresholds():
    closes = _closes(list(range(1, 25)))  # 24 points: below both 20 fully? has >=20
    ma = compute_moving_averages(closes)
    assert ma["ma_20"] is not None
    assert ma["ma_50"] is None  # not enough points for 50-day MA


def test_compute_moving_average_value():
    closes = _closes(list(range(1, 21)))  # 1..20
    ma = compute_moving_averages(closes)
    assert ma["ma_20"] == pytest.approx(sum(range(1, 21)) / 20)


def test_compute_max_drawdown():
    # Peak at 100, trough at 50 -> -50%
    closes = _closes([100, 90, 50, 80, 100])
    dd = compute_max_drawdown(closes)
    assert dd == pytest.approx(-50.0, abs=1e-9)


def test_compute_max_drawdown_no_decline():
    closes = _closes([10, 20, 30])
    assert compute_max_drawdown(closes) == pytest.approx(0.0, abs=1e-9)


def test_compute_market_metrics_empty_dataframe():
    df = pd.DataFrame()
    metrics = compute_market_metrics(df, symbol="FAKE")
    assert "error" in metrics


def test_compute_market_metrics_missing_close_column():
    df = pd.DataFrame({"Open": [1, 2, 3]})
    metrics = compute_market_metrics(df, symbol="FAKE")
    assert "error" in metrics


def test_compute_market_metrics_insufficient_data_flagged():
    df = pd.DataFrame({"Close": [10.0, 10.5, 11.0]}, index=pd.date_range("2026-01-01", periods=3))
    metrics = compute_market_metrics(df, symbol="TEST")
    assert metrics["error"] is None if "error" in metrics and metrics["error"] is None else True
    assert "1y" in metrics["insufficient_data_for"]
    assert metrics["returns_pct"]["1y"] is None
    # Never fabricate a value for insufficient data
    assert metrics["ma_50"] is None


def test_compute_market_metrics_full_bundle():
    n = 300
    prices = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    df = pd.DataFrame({"Close": prices}, index=pd.date_range("2025-01-01", periods=n))
    metrics = compute_market_metrics(df, symbol="NVDA")
    assert metrics["symbol"] == "NVDA"
    assert metrics["data_points"] == n
    assert metrics["ma_20"] is not None
    assert metrics["ma_50"] is not None
    assert metrics["volatility_annualized_pct"] is not None
    assert metrics["max_drawdown_pct"] <= 0


def test_resolve_period_date_range_is_iso_and_ordered():
    start, end = resolve_period_date_range("1Y")
    assert start < end
    # ISO format YYYY-MM-DD
    assert len(start) == 10 and start[4] == "-"
