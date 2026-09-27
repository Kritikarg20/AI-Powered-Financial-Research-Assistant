# Project Flow

This document walks through what actually happens, end to end, for each
feature — matching the current implementation (not any earlier design).

## 1. Document Research

1. User uploads one or more files (PDF, CSV, JSON, HTML, DOCX) on the
   **Document Research** page.
2. For each file, `fin_ai.core.rag.process_document()`:
   - extracts text (page-aware for PDF),
   - chunks it into ~1000-character pieces with paragraph-aware splitting,
   - embeds every chunk locally with `fastembed`,
   - writes a FAISS index (`vector_db/<doc_id>/index.faiss`) and a metadata
     JSON file (`vector_db/<doc_id>/metadata.json`) recording each chunk's
     source filename and page number.
3. The user selects which indexed documents to search (default: all) and
   asks a question.
4. `fin_ai.core.qa.answer_question()`:
   - embeds the question,
   - searches every selected document's FAISS index and merges the
     results by similarity score (this is the multi-document step),
   - sends the retrieved excerpts — each labeled with its source and page
     — to Mistral with an instruction to answer only from that evidence
     and to say so explicitly if it's insufficient.
5. The page displays the **Answer** and, separately, the **Source
   Evidence** (each retrieved excerpt in an expandable panel, with its
   source, page, and similarity score).

## 2. Market Analysis

1. User enters a ticker and selects a period.
2. `fin_ai.core.market.get_price_history()` fetches OHLC data (live via
   `yfinance`, or from bundled offline sample data if
   `YAHOO_SERVICE_OFFLINE=true`), with friendly errors for an invalid
   ticker, no data, or a network failure.
3. `fin_ai.core.metrics.compute_market_metrics()` computes, purely in
   Python: 1W/1M/3M/6M/1Y returns, annualized volatility, 20-day and
   50-day moving averages, and max drawdown. Any metric that needs more
   history than is available is returned as `None` and flagged under
   `insufficient_data_for` — never estimated or guessed.
4. The page renders a Plotly chart (close price + both moving averages)
   built from the same dataframe used for the metrics, plus a metrics
   panel.
5. If the user clicks **Interpret these metrics with Mistral**, the
   already-computed numbers (and only those numbers) are sent to Mistral
   with an explicit instruction not to recalculate or replace them —
   Mistral interprets, it does not compute.

## 3. Research Report

1. User provides a company/asset name, optionally selects documents and a
   ticker.
2. If a ticker is given, `market.get_metrics()` computes the same Python
   metrics as above.
3. If documents are selected, `report_pipeline.generate_research_report()`
   runs a handful of standard research queries (overview, financial
   performance, risks, strategy) against the selected documents' FAISS
   indexes and de-duplicates the retrieved evidence.
4. Evidence + metrics are formatted into one context block and sent to
   Mistral in a single prompt, with the required section structure
   (Executive Summary through Research Limitations / Sources) and
   explicit hallucination-control instructions (use only supplied
   evidence/metrics, never invent values or citations, flag missing
   evidence, distinguish fact from interpretation, not investment advice).
5. The generated Markdown report is displayed with a sources footer and
   is downloadable via the **Download report as Markdown** button.

## 4. Settings

Read-only: shows the configured LLM provider/model, whether an API key is
present (never the key itself), the embedding model and chunking
parameters, and whether market data is running in offline or live mode.
All of it is environment-driven (`.env`); this page does not write
configuration.
