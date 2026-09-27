"""Market Analysis — real market data, Python-computed metrics, AI interpretation."""

from dashboard.shared import render_sidebar_status
import plotly.graph_objects as go
import streamlit as st

from fin_ai.core.exceptions import LLMError
from fin_ai.core.llm import chat
from fin_ai.core.market import get_metrics, get_price_history

render_sidebar_status()

st.title("📈 Market Analysis")
st.caption(
    "Enter a ticker to see real market data, Python-computed financial metrics, "
    "and an AI interpretation of those metrics (Mistral never recalculates numbers)."
)

col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Ticker symbol", value="AAPL", placeholder="e.g. AAPL, MSFT, NVDA")
with col2:
    period = st.selectbox("Period", ["1M", "3M", "6M", "1Y", "2Y"], index=3)

if st.button("Analyze", type="primary", disabled=not symbol.strip()):
    with st.spinner(f"Fetching market data for {symbol.upper()}..."):
        df, error = get_price_history(symbol, period)

    if error:
        st.error(error)
    else:
        metrics = get_metrics(symbol, period)
        st.session_state["market_df"] = df
        st.session_state["market_metrics"] = metrics
        st.session_state["market_symbol"] = symbol.strip().upper()

df = st.session_state.get("market_df")
metrics = st.session_state.get("market_metrics")

if df is not None and metrics is not None and not metrics.get("error"):
    st.subheader(f"{metrics.get('symbol')} — Price Chart")

    close_col = "Close" if "Close" in df.columns else "close"
    ma20 = df[close_col].rolling(window=20).mean()
    ma50 = df[close_col].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[close_col], name="Close", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df.index, y=ma20, name="20-day MA", line=dict(color="#ff7f0e", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=ma50, name="50-day MA", line=dict(color="#2ca02c", dash="dot")))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Key Metrics (computed by Python)")
    returns = metrics.get("returns_pct", {}) or {}

    def _fmt_pct(v):
        return f"{v:.2f}%" if v is not None else "N/A"

    cols = st.columns(5)
    labels = [("1W", "1w"), ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("1Y", "1y")]
    for col, (label, key) in zip(cols, labels):
        col.metric(f"{label} Return", _fmt_pct(returns.get(key)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annualized Volatility", _fmt_pct(metrics.get("volatility_annualized_pct")))
    m2.metric("20-Day MA", f"{metrics['ma_20']:.2f}" if metrics.get("ma_20") is not None else "N/A")
    m3.metric("50-Day MA", f"{metrics['ma_50']:.2f}" if metrics.get("ma_50") is not None else "N/A")
    m4.metric("Max Drawdown", _fmt_pct(metrics.get("max_drawdown_pct")))

    if metrics.get("insufficient_data_for"):
        st.caption(
            "Insufficient historical data for: "
            + ", ".join(metrics["insufficient_data_for"])
            + ". These are shown as N/A above rather than estimated."
        )

    st.divider()
    st.subheader("AI Interpretation")
    if st.button("Interpret these metrics with Mistral"):
        with st.spinner("Asking Mistral to interpret the computed metrics..."):
            try:
                prompt = (
                    "The following financial metrics were computed by Python for "
                    f"ticker {metrics.get('symbol')}. Interpret them in plain English "
                    "for a research audience. Do not invent, replace, or recalculate "
                    "any numbers — use exactly the values given. If a value is N/A, "
                    "say the data was insufficient rather than guessing.\n\n"
                    f"1W Return: {_fmt_pct(returns.get('1w'))}\n"
                    f"1M Return: {_fmt_pct(returns.get('1m'))}\n"
                    f"3M Return: {_fmt_pct(returns.get('3m'))}\n"
                    f"6M Return: {_fmt_pct(returns.get('6m'))}\n"
                    f"1Y Return: {_fmt_pct(returns.get('1y'))}\n"
                    f"Annualized Volatility: {_fmt_pct(metrics.get('volatility_annualized_pct'))}\n"
                    f"20-Day MA: {metrics.get('ma_20')}\n"
                    f"50-Day MA: {metrics.get('ma_50')}\n"
                    f"Max Drawdown: {_fmt_pct(metrics.get('max_drawdown_pct'))}\n"
                )
                interpretation = chat(
                    [
                        {"role": "system", "content": "You are a financial research assistant, not a financial advisor."},
                        {"role": "user", "content": prompt},
                    ]
                )
                st.markdown(interpretation)
            except LLMError as exc:
                st.error(str(exc))
elif metrics and metrics.get("error"):
    st.error(metrics["error"])
