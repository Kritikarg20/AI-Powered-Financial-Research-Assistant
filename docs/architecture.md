# Architecture

## Overview

Financial Research AI is a Streamlit application with three research
features, all backed by the same simple building blocks: local embeddings +
FAISS for retrieval, Python for deterministic financial calculations, and
Mistral AI (via LiteLLM) for grounded generation and interpretation.

```
                    ┌─────────────────────┐
                    │     Streamlit UI     │
                    │       (app.py)       │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
 Document Research      Market Analysis       Research Report
          │                    │                     │
          ▼                    ▼                     ▼
      RAG / FAISS         Market Data          RAG Evidence
   (fastembed + faiss)   (yfinance/offline)     + Metrics
          │              Python Metrics              │
          ▼                    │                     │
      Mistral AI                ▼                     ▼
   (grounded answer)   Mistral Interpretation     Mistral AI
          │                                            │
          ▼                                            ▼
 Answer + Sources                              Final Report (.md)
```

## Components

### `dashboard/`
Five Streamlit pages (`home.py`, `document_research.py`,
`market_analysis.py`, `research_report.py`, `settings.py`) plus
`shared.py`, which renders a small sidebar status panel (Mistral
configured?, embedding model, document count, market data mode) used by
every page. `app.py` is the only entry point and the only place
`st.set_page_config()` / `st.navigation()` are called.

### `src/fin_ai/core/embeddings.py`
Wraps `fastembed` (a local ONNX-based embedding runtime) behind a small,
stable interface: `embed_documents(list[str])` and `embed_query(str)`. No
PyTorch/CUDA, no local server, no Ollama. The model file downloads once
from the Hugging Face Hub on first use and is then cached locally.

### `src/fin_ai/core/rag.py`
- Extracts text from PDF (PyMuPDF, page-aware), DOCX (python-docx), CSV
  (pandas), JSON, and HTML (BeautifulSoup).
- Chunks text with a simple, dependency-free paragraph-aware splitter.
- Embeds each chunk locally and stores it in a per-document FAISS
  (`IndexFlatIP`, cosine similarity via normalized vectors) index, with a
  parallel JSON metadata file recording source filename, page number, and
  chunk index.
- `search(query, doc_ids, top_k)` embeds the query, searches every
  requested document's index, and merges/ranks results by score — this is
  what makes multi-document retrieval work.

### `src/fin_ai/core/qa.py`
Retrieves chunks via `rag.search`, builds a context block that clearly
attributes each excerpt to its source and page, and sends a strict
grounding prompt to Mistral: answer only from the excerpts, say so
explicitly when evidence is insufficient, never invent citations.

### `src/fin_ai/core/service.py` + `metrics.py` + `market.py`
`service.py` fetches OHLC price history either live via `yfinance` or from
bundled offline sample data under `data/<TICKER>/`. `metrics.py` computes
returns (1W/1M/3M/6M/1Y), annualized volatility (`std(log returns) *
sqrt(252)`), 20/50-day moving averages, and max drawdown — all in Python,
never by the LLM — and reports `None`/insufficient-data explicitly rather
than fabricating a number. `market.py` wraps both with friendly error
handling so a bad ticker or network failure never surfaces a raw
traceback.

### `src/fin_ai/core/llm.py`
The only place that calls Mistral. Uses LiteLLM's `completion()` and maps
provider-specific exceptions (auth, rate limit, timeout, connection error)
to a single `LLMError` with a user-facing message.

### `src/fin_ai/core/report_pipeline.py`
A linear, non-agentic pipeline: pull RAG evidence across a handful of
standard research angles (overview, financial performance, risks,
strategy), format the already-computed market metrics, and send one
prompt to Mistral with explicit instructions not to recalculate or invent
numbers. Appends a sources footer listing every document/page actually
used.

## Why no AutoGen / multi-agent layer

The original codebase included a large AutoGen-based agent system. The
final architecture intentionally does not depend on it: the three core
features (document RAG, market analysis, research report) are each a
short, linear pipeline, and a multi-agent orchestration layer added
complexity without adding capability for this scope.

## Why no Ollama

The original codebase depended on a locally-running Ollama server for
embeddings. This requirement was removed entirely: embeddings are
generated in-process using `fastembed`, so the application has no external
server dependency for its GenAI/RAG functionality beyond the Mistral API
itself.
