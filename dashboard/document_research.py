"""Document Research — upload financial documents and ask grounded questions."""

import logging

from dashboard.shared import render_sidebar_status
import streamlit as st

from fin_ai.config.fin_ai import SUPPORTED_UPLOAD_TYPES
from fin_ai.core.exceptions import DocumentProcessingError, LLMError
from fin_ai.core.qa import answer_question
from fin_ai.core.rag import delete_document, list_documents, process_document

logger = logging.getLogger(__name__)

render_sidebar_status()

st.title("📄 Document Research")
st.caption(
    "Upload financial documents (PDF, CSV, JSON, HTML, DOCX). Ask questions and get "
    "answers grounded in the retrieved evidence, with source citations."
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more financial documents",
    type=SUPPORTED_UPLOAD_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded in uploaded_files:
        already_processed_key = f"processed_{uploaded.name}_{uploaded.size}"
        if st.session_state.get(already_processed_key):
            continue
        with st.spinner(f"Processing '{uploaded.name}'..."):
            try:
                data = uploaded.getvalue()
                info = process_document(uploaded.name, data)
                st.session_state[already_processed_key] = True
                st.success(
                    f"Indexed '{info['source']}' — {info['chunk_count']} chunk(s) "
                    f"embedded and stored."
                )
            except DocumentProcessingError as exc:
                st.error(f"Could not process '{uploaded.name}': {exc}")
            except Exception:
                logger.exception("Unexpected error while processing '%s'", uploaded.name)
                st.error(
                    f"Unexpected error while processing '{uploaded.name}'. "
                    "Please try a different file."
                )

st.divider()

# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------

documents = list_documents()

if not documents:
    st.info("No documents indexed yet. Upload a document above to get started.")
else:
    st.subheader("Indexed documents")
    doc_labels = {d["doc_id"]: f"{d['source']} ({d['chunk_count']} chunks)" for d in documents}

    col_select, col_delete = st.columns([3, 1])
    with col_select:
        selected_doc_ids = st.multiselect(
            "Select documents to search (multi-document retrieval)",
            options=list(doc_labels.keys()),
            default=list(doc_labels.keys()),
            format_func=lambda doc_id: doc_labels.get(doc_id, doc_id),
        )
    with col_delete:
        to_delete = st.selectbox("Remove a document", options=["-"] + list(doc_labels.keys()), format_func=lambda x: doc_labels.get(x, x))
        if st.button("Delete", disabled=(to_delete == "-")):
            delete_document(to_delete)
            st.rerun()

    st.divider()
    st.subheader("Ask a question")
    question = st.text_area("Your question", placeholder="e.g. What was revenue in the most recent quarter?")

    if st.button("Get answer", type="primary", disabled=not question.strip()):
        if not selected_doc_ids:
            st.warning("Select at least one document to search.")
        else:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                try:
                    result = answer_question(question, selected_doc_ids)
                    st.markdown("### Answer")
                    st.markdown(result.answer)

                    st.markdown("### Source Evidence")
                    if result.sources:
                        for i, chunk in enumerate(result.sources, start=1):
                            location = chunk.source + (f" — Page {chunk.page}" if chunk.page else "")
                            with st.expander(f"{i}. {location} (similarity {chunk.score:.2f})"):
                                st.write(chunk.text)
                    else:
                        st.caption("No source evidence was used for this answer.")
                except LLMError as exc:
                    st.error(str(exc))
                except DocumentProcessingError as exc:
                    st.error(str(exc))
                except Exception:
                    logger.exception("Unexpected error while answering a question")
                    st.error("Something went wrong while answering the question. Please try again.")
