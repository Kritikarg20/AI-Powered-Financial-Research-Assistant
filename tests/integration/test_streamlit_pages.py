"""
Streamlit AppTest integration tests.

Verifies each of the five required pages runs top-to-bottom without
raising an exception. These tests do not exercise external network calls
(Mistral API, live yfinance, embedding-model download) — they check that
the page *loads*: imports resolve, layout renders, and no code path
crashes on a cold start with no documents indexed and no API key
configured. Genuine external-API behaviour is covered separately by the
mocked unit tests for ``llm.py``, ``market.py``, and ``qa.py``.

AppTest runs each page script directly (not through ``app.py``'s
``st.navigation``), which is the supported way to test individual pages
of a multipage app with Streamlit's testing framework.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# AppTest executes each page script in-process using the current sys.path.
# The pages use `from dashboard.shared import ...`, which requires the
# project root (the parent of dashboard/) to be importable as a package
# root — exactly like it is when a person runs `streamlit run app.py`
# from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAGES = [
    "dashboard/home.py",
    "dashboard/document_research.py",
    "dashboard/market_analysis.py",
    "dashboard/research_report.py",
    "dashboard/settings.py",
]


@pytest.fixture(autouse=True)
def isolated_vector_db(tmp_path, monkeypatch):
    """Point the vector DB at an empty scratch directory for a clean-slate page load."""
    monkeypatch.setenv("VECTOR_DB_DIR", str(tmp_path))
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    yield


@pytest.mark.parametrize("page_path", PAGES)
def test_page_loads_without_exception(page_path):
    # Note: `st.page_link()` only validates cleanly inside a running
    # `st.navigation()` context (i.e. via `streamlit run app.py`). AppTest
    # runs a single page script directly, so `dashboard/home.py` wraps its
    # `st.page_link()` calls in a try/except fallback (see `_nav_link()` in
    # that file) specifically so it also renders cleanly here.
    at = AppTest.from_file(str(PROJECT_ROOT / page_path), default_timeout=30)
    at.run()
    assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"


def test_app_entrypoint_compiles():
    """app.py itself must at least be syntactically valid and importable-safe."""
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    compile(source, "app.py", "exec")
