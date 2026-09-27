# Financial Research AI

A student-level financial research assistant that combines **document RAG**,
**Mistral AI**, **market data**, and **deterministic Python financial
metrics** to help you research a company or asset.

This is an educational/research project. It does **not** place trades,
manage a portfolio, or give investment advice.

---

## What it does

- **Document Research** — upload financial documents (PDF, CSV, JSON, HTML,
  DOCX), ask questions, and get answers grounded in the retrieved evidence,
  with cited sources (document + page number where available). Works across
  multiple documents at once.
- **Market Analysis** — look up a ticker, see real price history, and view
  financial metrics computed by Python (returns, volatility, moving
  averages, max drawdown) with a Plotly chart. Mistral interprets the
  numbers in plain English — it never recalculates them.
- **Research Report** — combines retrieved document evidence with the
  Python-computed market metrics into a single structured prompt, and asks
  Mistral to generate a full research report you can download as Markdown.
- **Settings** — a read-only summary of the current configuration (no
  secrets displayed).

## Architecture

```
Document Research:
  Document -> extract text (+ page) -> chunk -> local embeddings (fastembed)
  -> FAISS index -> user question -> embed query -> retrieve top chunks
  -> Mistral (grounded answer) -> answer + cited sources

Market Analysis:
  Ticker -> market data (yfinance or bundled offline sample data)
  -> Python metrics (returns, volatility, moving averages, max drawdown)
  -> Plotly chart
  -> Mistral interpretation of the already-computed numbers

Research Report:
  RAG evidence + Python market metrics -> one structured prompt -> Mistral
  -> Markdown research report (download)
```

Mistral AI is the only LLM provider, called through **LiteLLM**. Embeddings
are generated **locally** with `fastembed` (a small ONNX model, no
PyTorch/CUDA) — there is **no Ollama, no local embedding server, and no
hosted embedding API** to configure. Vector search uses **FAISS**, one index
per processed document, so retrieval can be scoped to one or many documents
at once.

## Project structure

```
financial-research-ai/
├── app.py                     # streamlit run app.py
├── dashboard/                 # Streamlit pages
│   ├── shared.py               # sidebar status panel, shared imports
│   ├── home.py                 # Dashboard (landing page)
│   ├── document_research.py    # Document Research (RAG Q&A)
│   ├── market_analysis.py      # Market Analysis (metrics + chart + AI)
│   ├── research_report.py      # Research Report (RAG + metrics -> Mistral)
│   └── settings.py             # Settings (read-only config summary)
├── src/fin_ai/
│   ├── config/fin_ai.py        # all configuration (env-driven)
│   └── core/
│       ├── embeddings.py       # local embeddings (fastembed)
│       ├── rag.py              # extraction, chunking, FAISS store/search
│       ├── qa.py               # grounded document Q&A
│       ├── llm.py              # Mistral via LiteLLM, friendly errors
│       ├── service.py          # market data (online/offline)
│       ├── metrics.py          # Python-computed financial metrics
│       ├── market.py           # market data + metrics helper
│       ├── report_pipeline.py  # RAG + metrics -> Mistral -> report
│       └── exceptions.py
├── data/                      # bundled offline sample market data + docs
├── docs/                      # architecture & project-flow docs
├── tests/                     # pytest unit + integration tests
├── vector_db/                 # FAISS indexes + metadata (created at runtime)
├── published_research/        # saved reports (created at runtime)
├── requirements.in / .txt
└── .env.example
```

## Installation

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template and fill in your Mistral API key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env`:

```
MISTRAL_API_KEY=your-key-here
MISTRAL_MODEL=mistral-large-latest
```

Get a Mistral API key at [console.mistral.ai](https://console.mistral.ai/).

Run the app:

```bash
streamlit run app.py
```

No Ollama installation, no local server, and no GPU are required. The first
time you process a document, `fastembed` downloads a small embedding model
(a few hundred MB) from the Hugging Face Hub — after that it runs fully
offline/in-process.

## Market data modes

By default (`YAHOO_SERVICE_OFFLINE=true`) the app uses bundled sample data
under `data/<TICKER>/` so Market Analysis works with no internet access.
Set `YAHOO_SERVICE_OFFLINE=false` in `.env` to fetch live data via
`yfinance` instead.

## Testing

```bash
python -m compileall .
pip install -r requirements.txt
pytest
```

The test suite covers:

- Local embeddings (dimension, batching, empty input, missing-dependency error)
- RAG: extraction for PDF/CSV/JSON/HTML/DOCX, chunking, FAISS indexing,
  source+page metadata, **multi-document retrieval that distinguishes sources**
- Grounded Q&A (insufficient-evidence handling, no invented citations)
- Market metrics: returns, volatility, moving averages, max drawdown,
  insufficient-data handling — verified against manual/deterministic
  calculations
- Market data error handling (invalid ticker, network error, no raw
  traceback leaked)
- Mistral/LiteLLM error handling (missing key, auth error, rate limit,
  timeout, malformed response)
- The research report pipeline (evidence + metrics passed through
  correctly, no fabricated values)
- Streamlit `AppTest` — all five pages load without throwing an exception

Embedding-model downloads and live Mistral API calls require real internet
access and a valid API key; those specific network calls are mocked in the
test suite (see `tests/unit/fake_embedder.py` and the mocked `litellm`
module in `tests/unit/test_llm.py`) so the suite runs offline and
deterministically. See "Known limitations" below.

## Known limitations

- **Live Mistral completions are not exercised by the automated test
  suite** — they require a real `MISTRAL_API_KEY` and network access. The
  LLM wrapper's error handling (auth, rate limit, timeout, malformed
  response) is tested with a mocked LiteLLM client instead.
- **The local embedding model's actual download is not exercised by the
  automated test suite** in network-restricted environments — the wrapper
  logic is tested against a mocked backend. In normal use (with internet
  access) the model downloads automatically on first use.
- This is a research/educational tool, not a source of investment advice.

## Disclaimer

This project is for educational and research purposes only. It does not
constitute financial advice, and it does not place trades, manage a
portfolio, or execute any transactions.
