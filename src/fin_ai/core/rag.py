"""
RAG (Retrieval-Augmented Generation) core.

Pipeline::

    Document -> extract text (+ page/section info) -> chunk -> local
    embeddings -> FAISS index (one per document, metadata on disk)

    Query -> embed query -> search selected FAISS indexes -> merge/rank
    -> top-k chunks with source + page metadata

This module intentionally avoids any external vector-database server: each
processed document gets its own small FAISS index plus a metadata JSON file
under ``VECTOR_DB_DIR/<doc_id>/``. Multi-document retrieval works by
searching every selected document's index and merging the results by
similarity score, so answers can (and do) cite more than one source.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from fin_ai.config.fin_ai import CHUNK_OVERLAP, CHUNK_SIZE, VECTOR_DB_DIR
from fin_ai.core.embeddings import get_embedder
from fin_ai.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

VECTOR_DB_PATH = Path(VECTOR_DB_DIR)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    section: str | None
    score: float


# ---------------------------------------------------------------------------
# Text extraction — one function per supported format
# ---------------------------------------------------------------------------


def _extract_pdf(data: bytes) -> list[tuple[str, int | None]]:
    """Return a list of (page_text, page_number) tuples, 1-indexed pages."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentProcessingError(
            "PyMuPDF (pymupdf) is required to read PDF files."
        ) from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open PDF file: {exc}") from exc

    pages: list[tuple[str, int | None]] = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        text = text.strip()
        if text:
            pages.append((text, page_number))
    doc.close()
    if not pages:
        raise DocumentProcessingError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without a text layer."
        )
    return pages


def _extract_docx(data: bytes) -> list[tuple[str, int | None]]:
    try:
        import io

        import docx
    except ImportError as exc:
        raise DocumentProcessingError("python-docx is required to read .docx files.") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open Word document: {exc}") from exc

    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        raise DocumentProcessingError("No extractable text was found in this document.")
    return [(text, None)]


def _extract_csv(data: bytes) -> list[tuple[str, int | None]]:
    import io

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise DocumentProcessingError(f"Could not parse CSV file: {exc}") from exc
    if df.empty:
        raise DocumentProcessingError("The uploaded CSV file has no rows.")

    rows_per_chunk = 40
    sections: list[tuple[str, int | None]] = []
    for start in range(0, len(df), rows_per_chunk):
        block = df.iloc[start : start + rows_per_chunk]
        try:
            text = block.to_markdown(index=False)
        except Exception:
            text = block.to_string(index=False)
        header = f"Rows {start + 1}-{start + len(block)} of {len(df)}"
        sections.append((f"{header}\n{text}", None))
    return sections


def _extract_json(data: bytes) -> list[tuple[str, int | None]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise DocumentProcessingError(f"Could not parse JSON file: {exc}") from exc

    if isinstance(payload, list) and payload:
        chunks = []
        batch_size = 20
        for start in range(0, len(payload), batch_size):
            batch = payload[start : start + batch_size]
            chunks.append((json.dumps(batch, indent=2, default=str), None))
        return chunks
    text = json.dumps(payload, indent=2, default=str)
    if not text.strip():
        raise DocumentProcessingError("The uploaded JSON file is empty.")
    return [(text, None)]


def _extract_html(data: bytes) -> list[tuple[str, int | None]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise DocumentProcessingError("beautifulsoup4 is required to read HTML files.") from exc
    try:
        soup = BeautifulSoup(data, "html.parser")
    except Exception as exc:
        raise DocumentProcessingError(f"Could not parse HTML file: {exc}") from exc

    for tag in soup(["script", "style"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()
    if not text:
        raise DocumentProcessingError("No extractable text was found in this HTML file.")
    return [(text, None)]


_EXTRACTORS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "csv": _extract_csv,
    "json": _extract_json,
    "html": _extract_html,
    "htm": _extract_html,
}


def extract_text(filename: str, data: bytes) -> list[tuple[str, int | None]]:
    """Extract raw (text, page_number) sections from an uploaded file.

    Raises :class:`DocumentProcessingError` for unsupported or unreadable
    files. Never returns fabricated content.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        supported = ", ".join(sorted(set(_EXTRACTORS)))
        raise DocumentProcessingError(
            f"Unsupported file type '.{suffix}'. Supported types: {supported}."
        )
    if not data:
        raise DocumentProcessingError("The uploaded file is empty.")
    return extractor(data)


# ---------------------------------------------------------------------------
# Chunking — simple, dependency-free sliding window over characters,
# preferring paragraph/sentence boundaries where possible.
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Prefer splitting on paragraph breaks, falling back to raw characters.
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long — hard-split with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start = end - overlap if end - overlap > start else end
            current = ""
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def build_chunks(filename: str, sections: list[tuple[str, int | None]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for text, page in sections:
        for piece in chunk_text(text):
            chunks.append(Chunk(text=piece, source=filename, page=page, chunk_index=idx))
            idx += 1
    return chunks


# ---------------------------------------------------------------------------
# FAISS-backed store — one index per processed document
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return slug or uuid.uuid4().hex


def _doc_dir(doc_id: str) -> Path:
    return VECTOR_DB_PATH / doc_id


def process_document(filename: str, data: bytes) -> dict:
    """Extract, chunk, embed, and persist an uploaded document.

    Returns a summary dict: ``{"doc_id", "source", "chunk_count", "pages"}``.
    Raises :class:`DocumentProcessingError` on any failure — never silently
    produces an empty/fake index.
    """
    sections = extract_text(filename, data)
    chunks = build_chunks(filename, sections)
    if not chunks:
        raise DocumentProcessingError(f"No usable text chunks were extracted from '{filename}'.")

    embedder = get_embedder()
    try:
        vectors = embedder.embed_documents([c.text for c in chunks])
    except Exception as exc:
        raise DocumentProcessingError(f"Embedding generation failed: {exc}") from exc

    matrix = np.array(vectors, dtype="float32")
    dimension = matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized vectors
    index.add(matrix)

    doc_id = _slugify(filename)
    doc_dir = _doc_dir(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(doc_dir / "index.faiss"))
    metadata = [asdict(c) for c in chunks]
    with open(doc_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    pages = sorted({c.page for c in chunks if c.page is not None})
    info = {
        "doc_id": doc_id,
        "source": filename,
        "chunk_count": len(chunks),
        "embedding_model": embedder.model_name,
        "embedding_dim": dimension,
        "pages": pages,
    }
    with open(doc_dir / "info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    return info


def list_documents() -> list[dict]:
    """Return info dicts for every processed document currently on disk."""
    if not VECTOR_DB_PATH.exists():
        return []
    results = []
    for child in sorted(VECTOR_DB_PATH.iterdir()):
        info_path = child / "info.json"
        if info_path.is_file():
            try:
                results.append(json.loads(info_path.read_text(encoding="utf-8")))
            except Exception:
                # A corrupted/unreadable info.json shouldn't take the whole
                # list down, but silently dropping a document from the UI
                # with no trace makes "my document disappeared" undebuggable.
                logger.warning("Skipping unreadable document metadata at %s", info_path, exc_info=True)
                continue
    return results


def delete_document(doc_id: str) -> bool:
    doc_dir = _doc_dir(doc_id)
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
        return True
    return False


def _load_index_and_metadata(doc_id: str) -> tuple[Any, list[dict]]:
    doc_dir = _doc_dir(doc_id)
    index_path = doc_dir / "index.faiss"
    metadata_path = doc_dir / "metadata.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise DocumentProcessingError(f"No vector index found for document '{doc_id}'.")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata


def search(query: str, doc_ids: list[str], top_k: int = 5) -> list[RetrievedChunk]:
    """Search across one or more processed documents and return merged,
    score-ranked results (multi-document retrieval).
    """
    if not doc_ids:
        return []
    embedder = get_embedder()
    query_vector = np.array([embedder.embed_query(query)], dtype="float32")

    results: list[RetrievedChunk] = []
    for doc_id in doc_ids:
        try:
            index, metadata = _load_index_and_metadata(doc_id)
        except DocumentProcessingError:
            continue
        if index.ntotal == 0:
            continue
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vector, k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            row = metadata[idx]
            results.append(
                RetrievedChunk(
                    text=row["text"],
                    source=row["source"],
                    page=row.get("page"),
                    section=row.get("section"),
                    score=float(score),
                )
            )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
