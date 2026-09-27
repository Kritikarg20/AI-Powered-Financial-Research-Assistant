"""
Market data + metrics helper.

Thin convenience layer over :class:`fin_ai.core.service.MarketDataService`
and :mod:`fin_ai.core.metrics` that Streamlit pages call directly. All
error handling happens here so pages never see a raw traceback or a
yfinance/network exception.
"""

from __future__ import annotations

import logging

import pandas as pd

from fin_ai.core.exceptions import MarketDataNotFoundError, MarketDataServiceError
from fin_ai.core.metrics import compute_market_metrics, resolve_period_date_range
from fin_ai.core.service import MarketDataService

logger = logging.getLogger(__name__)

_service: MarketDataService | None = None


def get_service() -> MarketDataService:
    global _service
    if _service is None:
        _service = MarketDataService.from_environment()
    return _service


def get_price_history(symbol: str, period: str = "1Y") -> tuple[pd.DataFrame | None, str | None]:
    """Return (dataframe, error_message). Exactly one will be non-None."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None, "Please enter a ticker symbol."

    start, end = resolve_period_date_range(period)
    try:
        df = get_service().get_stock_data(symbol, start, end)
        return df, None
    except MarketDataNotFoundError:
        return None, (
            f"Unable to retrieve market data for '{symbol}'. "
            "Please verify the ticker symbol or try again later."
        )
    except MarketDataServiceError:
        return None, (
            "Unable to retrieve market data right now due to a network or "
            "data-provider issue. Please try again later."
        )
    except Exception:
        # Anything other than the two typed exceptions above is unexpected
        # (a real bug, not a "bad ticker" or "provider down" case) — log it
        # so it's debuggable, even though the user still gets the same
        # friendly, actionable message rather than a raw traceback.
        logger.exception("Unexpected error fetching market data for '%s'", symbol)
        return None, (
            "Unable to retrieve market data for this ticker. "
            "Please verify the ticker symbol or try again later."
        )


def get_metrics(symbol: str, period: str = "1Y") -> dict:
    """Return the full metrics bundle for *symbol*, or an ``{"error": ...}`` dict.

    Never raises — always returns a dict, so pages can uniformly check for
    an "error" key without wrapping every call in try/except.
    """
    df, error = get_price_history(symbol, period)
    if error:
        return {"symbol": symbol.strip().upper(), "error": error}
    metrics = compute_market_metrics(df, symbol=symbol.strip().upper())
    return metrics
