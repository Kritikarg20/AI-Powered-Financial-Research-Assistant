"""Unit tests for fin_ai.core.rag — extraction, chunking, FAISS retrieval."""

from __future__ import annotations

import json

import pytest

from fin_ai.core import rag as rag_module
from fin_ai.core.exceptions import DocumentProcessingError
from tests.unit.fake_embedder import FakeEmbedder


@pytest.fixture(autouse=True)
def fake_embedder_and_isolated_vector_db(tmp_path, monkeypatch):
    """Every test in this module uses the fake embedder and a scratch vector DB dir."""
    monkeypatch.setattr(rag_module, "get_embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(rag_module, "VECTOR_DB_PATH", tmp_path)
    yield


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_extract_csv():
    csv_bytes = b"ticker,price\nAAPL,150.25\nMSFT,380.75\n"
    sections = rag_module.extract_text("prices.csv", csv_bytes)
    assert len(sections) == 1
    text, page = sections[0]
    assert "AAPL" in text
    assert page is None


def test_extract_json_list():
    payload = [{"a": 1}, {"a": 2}]
    sections = rag_module.extract_text("data.json", json.dumps(payload).encode())
    assert sections
    assert "a" in sections[0][0]


def test_extract_html_strips_tags():
    html = b"<html><body><h1>Report</h1><p>Revenue grew 10%.</p></body></html>"
    sections = rag_module.extract_text("report.html", html)
    text, page = sections[0]
    assert "Revenue grew 10%" in text
    assert "<h1>" not in text


def test_extract_docx():
    import io

    import docx

    document = docx.Document()
    document.add_paragraph("Quarterly revenue increased significantly.")
    buf = io.BytesIO()
    document.save(buf)
    sections = rag_module.extract_text("report.docx", buf.getvalue())
    assert "Quarterly revenue" in sections[0][0]


def test_extract_pdf_with_page_numbers():
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one content about revenue.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two content about risk factors.")
    data = doc.tobytes()
    doc.close()

    sections = rag_module.extract_text("filing.pdf", data)
    assert len(sections) == 2
    assert sections[0][1] == 1
    assert sections[1][1] == 2
    assert "revenue" in sections[0][0].lower()
    assert "risk" in sections[1][0].lower()


def test_extract_unsupported_type_raises():
    with pytest.raises(DocumentProcessingError):
        rag_module.extract_text("file.xyz", b"data")


def test_extract_empty_file_raises():
    with pytest.raises(DocumentProcessingError):
        rag_module.extract_text("empty.csv", b"")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_short_text_single_chunk():
    chunks = rag_module.chunk_text("short text", chunk_size=1000)
    assert chunks == ["short text"]


def test_chunk_text_splits_long_text():
    long_text = "\n\n".join([f"Paragraph {i} " + ("word " * 50) for i in range(20)])
    chunks = rag_module.chunk_text(long_text, chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 330 for c in chunks)  # allow small slack from paragraph joins


def test_chunk_text_empty_returns_empty_list():
    assert rag_module.chunk_text("   ") == []


# ---------------------------------------------------------------------------
# End-to-end: process_document + search (single document)
# ---------------------------------------------------------------------------


def test_process_document_and_search_single_doc():
    csv_bytes = b"ticker,note\nAAPL,Apple reported strong iPhone revenue growth\nMSFT,Microsoft cloud Azure grew fast\n"
    info = rag_module.process_document("notes.csv", csv_bytes)
    assert info["chunk_count"] >= 1
    assert info["source"] == "notes.csv"

    results = rag_module.search("iPhone revenue growth", [info["doc_id"]], top_k=3)
    assert results
    assert all(r.source == "notes.csv" for r in results)


def test_process_document_raises_on_bad_input():
    with pytest.raises(DocumentProcessingError):
        rag_module.process_document("bad.csv", b"")


def test_list_and_delete_document():
    csv_bytes = b"a,b\n1,2\n"
    info = rag_module.process_document("x.csv", csv_bytes)
    docs = rag_module.list_documents()
    assert any(d["doc_id"] == info["doc_id"] for d in docs)

    assert rag_module.delete_document(info["doc_id"]) is True
    docs_after = rag_module.list_documents()
    assert not any(d["doc_id"] == info["doc_id"] for d in docs_after)


# ---------------------------------------------------------------------------
# Multi-document retrieval
# ---------------------------------------------------------------------------


def test_multi_document_retrieval_distinguishes_sources():
    annual_report = (
        b"Annual revenue increased due to strong cloud demand and enterprise contracts. "
        b"Full year net income rose year over year."
    )
    quarterly_report = (
        b"Quarterly earnings call highlighted supply chain risk and rising input costs. "
        b"Management flagged inflation as a key risk factor for the quarter."
    )
    info_a = rag_module.process_document("annual_report.txt".replace(".txt", ".html"), b"<p>" + annual_report + b"</p>")
    info_b = rag_module.process_document("quarterly_report.html", b"<p>" + quarterly_report + b"</p>")

    all_docs = [info_a["doc_id"], info_b["doc_id"]]

    revenue_results = rag_module.search("cloud revenue and enterprise demand", all_docs, top_k=5)
    assert revenue_results
    assert revenue_results[0].source == "annual_report.html"

    risk_results = rag_module.search("supply chain risk and inflation", all_docs, top_k=5)
    assert risk_results
    assert risk_results[0].source == "quarterly_report.html"

    # Sanity: searching only one of the two documents never returns the other's source.
    only_annual = rag_module.search("inflation risk", [info_a["doc_id"]], top_k=5)
    assert all(r.source == "annual_report.html" for r in only_annual)


def test_search_with_no_doc_ids_returns_empty():
    assert rag_module.search("anything", [], top_k=5) == []


def test_search_with_unknown_doc_id_returns_empty():
    assert rag_module.search("anything", ["does-not-exist"], top_k=5) == []
