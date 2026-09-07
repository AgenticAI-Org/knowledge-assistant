"""Streamlit UI for the Employee Knowledge Assistant.

Run: uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# `streamlit run` executes this file directly (not as a package), so the
# project root needs to be added to sys.path for `from src...` imports to work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import settings
from src.generation.answer_service import answer as run_answer
from src.ingestion.indexer import build_index
from src.logging_config import configure_logging
from src.memory.conversation import ChatTurn
from src.schemas import Answer

configure_logging()

st.set_page_config(page_title="Employee Knowledge Assistant", page_icon="📚", layout="centered")


def _index_exists() -> bool:
    return settings.chroma_persist_dir.exists() and settings.bm25_index_path.exists()


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{"role": "user"|"assistant", "content": str, "answer": Answer|None}]
    if "history" not in st.session_state:
        st.session_state.history = []  # list[ChatTurn], the conversational memory


def _render_sources(answer_obj: Answer) -> None:
    if not answer_obj.citations:
        return
    with st.expander(f"Sources ({len(answer_obj.citations)})"):
        for citation in answer_obj.citations:
            st.markdown(f"**{citation.source}** — {len(citation.chunk_ids)} excerpt(s) used")
        if answer_obj.retrieved_chunks:
            st.caption(f"Answered in {answer_obj.latency_ms} ms")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Employee Knowledge Assistant")
        st.caption("Ask about leave, IT, travel, benefits, and code-of-conduct policies.")

        st.divider()
        st.subheader("Knowledge base")
        if _index_exists():
            st.success("Index is built and ready.")
        else:
            st.warning("No index found yet. Build it before asking questions.")

        if st.button("🔄 (Re)build index from data/", use_container_width=True):
            with st.spinner("Loading documents, chunking, embedding, and indexing..."):
                try:
                    stats = build_index()
                except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                    st.error(f"Index build failed: {exc}")
                else:
                    st.success(f"Indexed {stats.num_chunks} chunks from {stats.num_source_documents} documents.")
                    st.rerun()

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("answer") is not None:
                _render_sources(message["answer"])


def handle_user_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question, "answer": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not _index_exists():
            error_text = "The knowledge base hasn't been indexed yet. Use the sidebar to build the index first."
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "answer": None})
            return

        with st.spinner("Thinking..."):
            try:
                answer_obj = run_answer(
                    question=question,
                    history=st.session_state.history,
                    session_id=st.session_state.session_id,
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                error_text = f"Something went wrong while answering: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "answer": None})
                return

        st.markdown(answer_obj.text)
        _render_sources(answer_obj)

    st.session_state.messages.append({"role": "assistant", "content": answer_obj.text, "answer": answer_obj})
    if answer_obj.grounded:
        st.session_state.history.append(ChatTurn(question=question, answer=answer_obj.text))


def main() -> None:
    _init_session_state()
    render_sidebar()

    st.title("📚 Employee Knowledge Assistant")
    render_chat_history()

    question = st.chat_input("Ask a question about company policies...")
    if question:
        handle_user_question(question)


if __name__ == "__main__":
    main()
