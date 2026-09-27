"""
Quantitative market metrics — computed in Python, never by the LLM.

This module is the "Python calculates, Mistral interprets" half of the
market-analysis feature. It takes the raw OHLCV DataFrame already returned
by ``MarketDataService.get_stock_data`` (see ``fin_ai.core.market.get_service``)
and derives the numbers a research report needs: returns over several
windows, historical volatility, moving averages, and maximum drawdown.

Usage::

    from fin_ai.core.market import get_service
    from fin_ai.core.metrics import compute_market_metrics

    df = get_service().get_stock_data("NVDA", "2025-01-01", "2026-09-01")
    metrics = compute_market_metrics(df)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Approximate calendar-day windows for each selectable analysis period.
PERIOD_TO_DAYS = {
    "1M": 31,
    "3M": 93,
    "6M": 186,
    "1Y": 366,
    "2Y": 732,
}


def resolve_period_date_range(period: str) -> tuple[str, str]:
    """Convert a period label (e.g. '1Y') into (start_date, end_date) ISO strings."""
    days = PERIOD_TO_DAYS.get(period.upper(), 366)
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


# Approximate trading-day windows for each named return period.
RETURN_WINDOWS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}


def _pct_change_over(closes: pd.Series, periods: int) -> float | None:
    """Percent change from `periods` trading days ago to the last close.

    Returns None if there isn't enough history for the requested window.
    """
    if len(closes) <= periods:
        return None
    start_price = closes.iloc[-(periods + 1)]
    end_price = closes.iloc[-1]
    if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
        return None
    return float((end_price / start_price - 1) * 100)


def compute_returns(closes: pd.Series) -> dict[str, float | None]:
    """Percent return for each window in RETURN_WINDOWS, in percent (e.g. 8.4 = +8.4%)."""
    return {label: _pct_change_over(closes, window) for label, window in RETURN_WINDOWS.items()}


def compute_volatility(closes: pd.Series, annualize: bool = True) -> float | None:
    """Historical volatility of daily returns, in percent.

    Standard deviation of daily log returns, annualized by default
    (multiplied by sqrt(252)).
    """
    if len(closes) < 2:
        return None
    daily_returns = np.log(closes / closes.shift(1)).dropna()
    if daily_returns.empty:
        return None
    vol = daily_returns.std()
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(vol * 100)


def compute_moving_averages(closes: pd.Series) -> dict[str, float | None]:
    """Latest 20-day and 50-day simple moving averages."""
    result: dict[str, float | None] = {"ma_20": None, "ma_50": None}
    if len(closes) >= 20:
        result["ma_20"] = float(closes.rolling(window=20).mean().iloc[-1])
    if len(closes) >= 50:
        result["ma_50"] = float(closes.rolling(window=50).mean().iloc[-1])
    return result


def compute_max_drawdown(closes: pd.Series) -> float | None:
    """Maximum peak-to-trough decline over the series, in percent (negative number)."""
    if closes.empty:
        return None
    running_max = closes.cummax()
    drawdown = (closes - running_max) / running_max
    return float(drawdown.min() * 100)


def compute_market_metrics(df: pd.DataFrame, symbol: str | None = None) -> dict:
    """Compute the full metrics bundle for a research report or chart.

    Parameters
    ----------
    df : DataFrame
        OHLCV data as returned by ``MarketDataService.get_stock_data`` —
        must have a ``Close`` column (case-insensitive fallback to ``close``).
    symbol : str, optional
        Included in the output for convenience/logging.

    Returns
    -------
    dict with keys: symbol, latest_price, open, high, low, close,
    returns (dict), volatility_annualized_pct, ma_20, ma_50,
    max_drawdown_pct, data_points, insufficient_data (list of flags).
    """
    if df is None or df.empty:
        return {
            "symbol": symbol,
            "error": "No price data available for this ticker/period.",
        }

    close_col = "Close" if "Close" in df.columns else "close"
    if close_col not in df.columns:
        return {"symbol": symbol, "error": "Price data has no Close column."}

    closes = df[close_col].dropna()
    latest = df.iloc[-1]

    returns = compute_returns(closes)
    moving_avgs = compute_moving_averages(closes)
    volatility = compute_volatility(closes)
    max_dd = compute_max_drawdown(closes)

    insufficient = [
        label for label, window in RETURN_WINDOWS.items()
        if returns.get(label) is None
    ]

    def _get(col_options: list[str]):
        for c in col_options:
            if c in df.columns:
                return float(latest[c])
        return None

    return {
        "symbol": symbol,
        "latest_price": float(closes.iloc[-1]),
        "open": _get(["Open", "open"]),
        "high": _get(["High", "high"]),
        "low": _get(["Low", "low"]),
        "close": float(closes.iloc[-1]),
        "returns_pct": returns,
        "volatility_annualized_pct": volatility,
        "ma_20": moving_avgs["ma_20"],
        "ma_50": moving_avgs["ma_50"],
        "max_drawdown_pct": max_dd,
        "data_points": int(len(closes)),
        "insufficient_data_for": insufficient,
    }
