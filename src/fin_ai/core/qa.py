"""
Grounded document Q&A — the core of the Document Research page.

    User question -> embed -> FAISS search (one or more documents)
    -> retrieved chunks -> Mistral (grounded generation) -> answer + sources

The LLM is explicitly instructed to answer only from the supplied
evidence and to say so when the evidence is insufficient, so citations are
never invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fin_ai.config.fin_ai import RETRIEVAL_TOP_K
from fin_ai.core.exceptions import DocumentProcessingError
from fin_ai.core.llm import chat
from fin_ai.core.rag import RetrievedChunk, search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's \
question using ONLY the provided document excerpts as evidence.

Rules:
- Do not use outside knowledge or invent any facts, figures, or citations.
- Every factual claim in your answer must be supported by the excerpts below.
- If the excerpts do not contain enough information to answer, say clearly: \
"The provided documents do not contain enough information to answer this question."
- Do not fabricate page numbers or document names beyond what is given.
- Keep the answer concise and focused on the question."""


@dataclass
class QAResult:
    answer: str
    sources: list[RetrievedChunk]
    insufficient_evidence: bool


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        location = f"{c.source}" + (f", page {c.page}" if c.page else "")
        parts.append(f"[Excerpt {i} — {location}]\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str, doc_ids: list[str], top_k: int = RETRIEVAL_TOP_K) -> QAResult:
    """Run the full retrieve -> ground -> generate pipeline for *question*."""
    question = (question or "").strip()
    if not question:
        return QAResult(answer="Please enter a question.", sources=[], insufficient_evidence=True)
    if not doc_ids:
        return QAResult(
            answer="No documents are selected. Upload and select at least one document first.",
            sources=[],
            insufficient_evidence=True,
        )

    try:
        chunks = search(question, doc_ids, top_k=top_k)
    except Exception as exc:
        # search() embeds the query with the local embedder before it ever
        # touches FAISS, so a failure here is almost always "the embedding
        # model couldn't be loaded" (e.g. no internet on first use) rather
        # than a FAISS/index problem. Log the real cause and surface a
        # specific, actionable message instead of letting a raw RuntimeError
        # fall through to the page's generic catch-all.
        logger.exception("Retrieval failed while answering a question")
        raise DocumentProcessingError(
            f"Could not search the selected documents: {exc}"
        ) from exc
    if not chunks:
        return QAResult(
            answer="The provided documents do not contain enough information to answer this question.",
            sources=[],
            insufficient_evidence=True,
        )

    context = _format_context(chunks)
    user_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    answer = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    insufficient = "do not contain enough information" in answer.lower()
    return QAResult(answer=answer, sources=chunks, insufficient_evidence=insufficient)
