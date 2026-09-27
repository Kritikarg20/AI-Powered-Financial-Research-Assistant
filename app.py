# streamlit run app.py
"""
Financial Research AI — application entry point.

This is the only place ``st.set_page_config()`` is called (Streamlit
requires it be called exactly once, before any other Streamlit command).
Everything else — the sidebar, and each individual page's content — lives
under ``dashboard/``:

    dashboard/shared.py             shared sidebar + config, used by every page
    dashboard/home.py                Dashboard (landing page)
    dashboard/document_research.py  Document Research (RAG Q&A with cited sources)
    dashboard/market_analysis.py    Market Analysis (metrics, chart, AI interpretation)
    dashboard/research_report.py    Research Report (combined RAG + market pipeline)
    dashboard/settings.py           Settings (read-only config summary)

Run with:

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Financial Research AI", layout="wide", page_icon="💹")

pages = [
    st.Page("dashboard/home.py", title="Dashboard", icon="🏠", default=True),
    st.Page("dashboard/document_research.py", title="Document Research", icon="📄"),
    st.Page("dashboard/market_analysis.py", title="Market Analysis", icon="📈"),
    st.Page("dashboard/research_report.py", title="Research Report", icon="📑"),
    st.Page("dashboard/settings.py", title="Settings", icon="⚙️"),
]

pg = st.navigation(pages)
pg.run()
