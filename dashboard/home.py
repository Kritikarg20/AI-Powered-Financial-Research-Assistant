"""Dashboard — landing page overview."""

from dashboard.shared import render_sidebar_status
import streamlit as st

from fin_ai.core.rag import list_documents

render_sidebar_status()

st.title("💹 Financial Research AI")
st.caption("A student research assistant combining document RAG, market data, and Mistral AI.")

st.markdown(
    """
This project helps you research a company or asset by combining:

- **Financial documents** you upload (annual reports, filings, transcripts, spreadsheets)
- **Market data** (price history, returns, volatility)
- **Mistral AI**, used to interpret data and generate grounded, cited answers and reports

It is an educational/research tool. It does not place trades, manage a
portfolio, or give investment advice.
"""
)

st.divider()


def _nav_link(path: str, label: str) -> None:
    """st.page_link() only works inside a running st.navigation() context
    (i.e. via `streamlit run app.py`). Fall back to plain text so this page
    still renders cleanly if ever loaded outside that context (e.g. a
    testing harness that runs this script directly)."""
    try:
        st.page_link(path, label=label)
    except Exception:
        st.caption(label)


col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("📄 Document Research")
    st.write("Upload financial documents and ask grounded questions with cited sources.")
    _nav_link("dashboard/document_research.py", "Go to Document Research →")

with col2:
    st.subheader("📈 Market Analysis")
    st.write("Look up a ticker, see Python-computed metrics, and get AI interpretation.")
    _nav_link("dashboard/market_analysis.py", "Go to Market Analysis →")

with col3:
    st.subheader("📑 Research Report")
    st.write("Combine document evidence and market metrics into one Mistral-generated report.")
    _nav_link("dashboard/research_report.py", "Go to Research Report →")

st.divider()

st.subheader("Document status")
docs = list_documents()
if docs:
    st.dataframe(
        [
            {
                "Document": d.get("source"),
                "Chunks": d.get("chunk_count"),
                "Pages": len(d.get("pages", []) or []),
            }
            for d in docs
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No documents indexed yet. Upload one on the Document Research page to get started.")
