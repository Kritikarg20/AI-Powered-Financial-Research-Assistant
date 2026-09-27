"""Research Report — combines document evidence and market metrics into one report."""

import logging

from dashboard.shared import render_sidebar_status
import streamlit as st

from fin_ai.core.exceptions import DocumentProcessingError, LLMError
from fin_ai.core.market import get_metrics
from fin_ai.core.rag import list_documents
from fin_ai.core.report_pipeline import generate_research_report

logger = logging.getLogger(__name__)

render_sidebar_status()

st.title("📑 Research Report")
st.caption(
    "Combines retrieved document evidence with Python-computed market metrics, "
    "then asks Mistral to synthesize a structured research report."
)

documents = list_documents()
doc_labels = {d["doc_id"]: f"{d['source']} ({d['chunk_count']} chunks)" for d in documents}

col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Company / asset name", placeholder="e.g. NVIDIA")
with col2:
    ticker = st.text_input("Ticker (optional, for market metrics)", placeholder="e.g. NVDA")

selected_doc_ids = st.multiselect(
    "Documents to use as evidence (optional)",
    options=list(doc_labels.keys()),
    default=list(doc_labels.keys()),
    format_func=lambda doc_id: doc_labels.get(doc_id, doc_id),
)

period = st.selectbox("Market data period", ["1M", "3M", "6M", "1Y", "2Y"], index=3, disabled=not ticker.strip())

if st.button("Generate report", type="primary", disabled=not subject.strip()):
    metrics = None
    if ticker.strip():
        with st.spinner(f"Fetching market metrics for {ticker.upper()}..."):
            metrics = get_metrics(ticker, period)
            if metrics.get("error"):
                st.warning(f"Market metrics unavailable: {metrics['error']} The report will note this limitation.")

    if not selected_doc_ids and not ticker.strip():
        st.warning("Select at least one document or provide a ticker so the report has evidence to work with.")
    else:
        with st.spinner("Retrieving evidence, computing metrics, and generating the report with Mistral..."):
            try:
                report_md = generate_research_report(subject, doc_ids=selected_doc_ids, metrics=metrics)
                st.session_state["last_report"] = report_md
                st.session_state["last_report_subject"] = subject
            except LLMError as exc:
                st.error(str(exc))
            except DocumentProcessingError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Unexpected error while generating the report")
                st.error("Something went wrong while generating the report. Please try again.")

report_md = st.session_state.get("last_report")
if report_md:
    st.divider()
    st.markdown(report_md)
    st.download_button(
        "Download report as Markdown",
        data=report_md,
        file_name=f"{st.session_state.get('last_report_subject', 'research_report').replace(' ', '_')}.md",
        mime="text/markdown",
    )
