"""
Research report pipeline.

    Financial documents (RAG evidence) + Python market metrics
        -> one structured prompt -> Mistral -> Markdown research report

Deliberately a simple, linear pipeline (no multi-agent orchestration):

    1. Retrieve relevant document evidence for the given topic/company.
    2. Read the already-computed market metrics (Python, not the LLM).
    3. Build one structured context block.
    4. Send a single well-designed prompt to Mistral.
    5. Return the generated Markdown report.
"""

from __future__ import annotations

import logging
from datetime import date

from fin_ai.core.exceptions import DocumentProcessingError
from fin_ai.core.llm import chat
from fin_ai.core.rag import search

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are a financial research analyst writing an \
educational research report. Follow these rules strictly:

- Use ONLY the supplied document evidence and the supplied market metrics.
- Market metrics are pre-computed by Python and must be repeated exactly as \
given — never recalculate, round differently, or invent a different number.
- Do not invent financial facts, figures, or citations that are not present \
in the evidence provided.
- Clearly distinguish between facts drawn from the documents/metrics and your \
own interpretation or analysis.
- If evidence for a section is not available, say so explicitly in that \
section instead of making something up.
- This is a research/educational report, NOT investment advice. Do not \
present conclusions as guaranteed outcomes or a recommendation to buy/sell.
- Write in clean Markdown with clear section headings."""

REPORT_STRUCTURE = """Structure the report with exactly these sections:

1. Executive Summary
2. Company / Asset Overview
3. Financial Analysis
4. Key Financial Metrics
5. Market Performance
6. Key Drivers / Catalysts
7. Risks & Red Flags
8. AI-Based Interpretation
9. Research Limitations / Sources"""


def _format_evidence(chunks) -> str:
    if not chunks:
        return "No relevant document evidence was retrieved."
    parts = []
    for i, c in enumerate(chunks, start=1):
        location = f"{c.source}" + (f", page {c.page}" if c.page else "")
        parts.append(f"[Evidence {i} — {location}]\n{c.text}")
    return "\n\n".join(parts)


def _format_metrics(metrics: dict) -> str:
    if not metrics or metrics.get("error"):
        return f"No market metrics available. Reason: {metrics.get('error') if metrics else 'not requested'}"

    lines = [f"Ticker: {metrics.get('symbol', 'N/A')}"]
    returns = metrics.get("returns_pct", {}) or {}
    for label, key in [("1W", "1w"), ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("1Y", "1y")]:
        value = returns.get(key)
        lines.append(f"{label} Return: {f'{value:.2f}%' if value is not None else 'N/A (insufficient historical data)'}")
    vol = metrics.get("volatility_annualized_pct")
    lines.append(f"Annualized Volatility: {f'{vol:.2f}%' if vol is not None else 'N/A'}")
    ma20 = metrics.get("ma_20")
    ma50 = metrics.get("ma_50")
    lines.append(f"20-Day Moving Average: {f'{ma20:.2f}' if ma20 is not None else 'N/A'}")
    lines.append(f"50-Day Moving Average: {f'{ma50:.2f}' if ma50 is not None else 'N/A'}")
    dd = metrics.get("max_drawdown_pct")
    lines.append(f"Maximum Drawdown: {f'{dd:.2f}%' if dd is not None else 'N/A'}")
    lines.append(f"Latest Price: {metrics.get('latest_price', 'N/A')}")
    return "\n".join(lines)


def generate_research_report(
    subject: str,
    doc_ids: list[str] | None = None,
    metrics: dict | None = None,
) -> str:
    """Generate a structured Markdown research report for *subject*.

    Parameters
    ----------
    subject : str
        Company name or ticker the report is about.
    doc_ids : list[str], optional
        Processed document ids to pull RAG evidence from.
    metrics : dict, optional
        Output of :func:`fin_ai.core.market.get_metrics`.
    """
    doc_ids = doc_ids or []
    evidence_chunks = []
    if doc_ids:
        # Pull evidence for a handful of standard research angles so the
        # report has broad coverage rather than just the top match for one query.
        queries = [
            subject,
            f"{subject} revenue and financial performance",
            f"{subject} risks and challenges",
            f"{subject} business overview and strategy",
        ]
        seen = set()
        try:
            for q in queries:
                for chunk in search(q, doc_ids, top_k=4):
                    key = (chunk.source, chunk.page, chunk.text[:50])
                    if key not in seen:
                        seen.add(key)
                        evidence_chunks.append(chunk)
        except Exception as exc:
            # Same failure mode as fin_ai.core.qa: search() embeds the query
            # before touching FAISS, so this is almost always the local
            # embedder failing to load. Log the real cause and surface a
            # specific message rather than a generic page-level fallback.
            logger.exception("Retrieval failed while gathering report evidence")
            raise DocumentProcessingError(
                f"Could not retrieve document evidence for '{subject}': {exc}"
            ) from exc

    context = (
        f"Report subject: {subject}\n"
        f"Report date: {date.today().isoformat()}\n\n"
        f"=== Document Evidence ===\n{_format_evidence(evidence_chunks)}\n\n"
        f"=== Market Metrics (computed by Python — do not recalculate) ===\n"
        f"{_format_metrics(metrics or {})}\n"
    )

    user_prompt = f"{context}\n\n{REPORT_STRUCTURE}\n\nWrite the full research report now."

    report = chat(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3000,
    )

    sources_note = _build_sources_footer(evidence_chunks, metrics)
    return f"{report}\n\n{sources_note}"


def _build_sources_footer(evidence_chunks, metrics: dict | None) -> str:
    lines = ["---", "**Sources used to generate this report:**"]
    if evidence_chunks:
        seen_sources = sorted({(c.source, c.page) for c in evidence_chunks}, key=lambda t: (t[0], t[1] or 0))
        for source, page in seen_sources:
            lines.append(f"- {source}" + (f" — page {page}" if page else ""))
    else:
        lines.append("- No document evidence was used.")
    if metrics and not metrics.get("error"):
        lines.append(f"- Market data for {metrics.get('symbol', 'N/A')} (computed metrics, not LLM-generated)")
    lines.append("")
    lines.append(
        "*This report is generated for research/educational purposes only and "
        "is not financial advice.*"
    )
    return "\n".join(lines)
