from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_runtime_config
from app.graph import build_graph
from app.graph.state import create_initial_state
from app.observability import (
    InMemoryLogHandler,
    configure_logging,
    get_logger,
    truncate_text,
)

logger = get_logger("ui.streamlit")


def _bootstrap_app() -> None:
    configure_logging()
    load_runtime_config(
        config_path=PROJECT_ROOT / "config.yml",
        env_path=PROJECT_ROOT / ".env",
    )


@st.cache_resource(show_spinner=False)
def _get_graph():
    _bootstrap_app()
    return build_graph()


def _ensure_session_state() -> None:
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = create_initial_state("")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello, I'm your customer care demo assistant. "
                    "Ask a question or start an appointment request."
                ),
                "retrieval_query": "",
                "retrieved_context": [],
                "turn_logs": [],
            }
        ]

    if "trace_handler" not in st.session_state:
        trace_handler = InMemoryLogHandler()
        configure_logging().addHandler(trace_handler)
        st.session_state.trace_handler = trace_handler

    if "turn_logs" not in st.session_state:
        st.session_state.turn_logs = []


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("Customer Care Demo")
        st.caption("Standalone Streamlit UI for the existing LangGraph backend.")

        if st.button("Reset Chat", use_container_width=True):
            st.session_state.chat_state = create_initial_state("")
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Chat reset. How can I help you today?"
                    ),
                    "retrieval_query": "",
                    "retrieved_context": [],
                    "turn_logs": [],
                }
            ]
            st.rerun()

        current_state = st.session_state.chat_state
        st.divider()
        st.subheader("Session State")
        st.write(
            {
                "intent": current_state.get("intent"),
                "active_action": current_state.get("active_action"),
                "handoff_pending": current_state.get("handoff_pending"),
                "failure_count": current_state.get("failure_count"),
                "awaiting_confirmation": current_state.get("awaiting_confirmation"),
            }
        )
        st.divider()
        st.subheader("Vector Query")
        retrieval_query = str(current_state.get("retrieval_query", "")).strip()
        if retrieval_query:
            st.code(retrieval_query, language="text")
        else:
            st.caption("No vector query was used on the current turn.")

        st.divider()
        st.subheader("Retrieved Context")
        retrieved_context = current_state.get("retrieved_context", [])
        if retrieved_context:
            for index, item in enumerate(retrieved_context, start=1):
                st.caption(f"Chunk {index}")
                st.code(item, language="text")
        else:
            st.caption("No chunks retrieved on the current turn.")

        st.divider()
        st.subheader("Backend Trace")
        turn_logs = st.session_state.turn_logs
        if turn_logs:
            st.code("\n".join(turn_logs), language="text")
        else:
            st.caption("Logs will appear here after the first message.")


def _render_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))

        retrieval_query = str(message.get("retrieval_query", "")).strip()
        if retrieval_query:
            with st.expander("Vector DB Query", expanded=False):
                st.code(retrieval_query, language="text")

        retrieved_context = message.get("retrieved_context", [])
        if isinstance(retrieved_context, list) and retrieved_context:
            with st.expander(
                f"Retrieved Chunks ({len(retrieved_context)})",
                expanded=False,
            ):
                for index, item in enumerate(retrieved_context, start=1):
                    st.caption(f"Chunk {index}")
                    st.code(str(item), language="text")


def _run_turn(user_query: str) -> dict[str, object]:
    trace_handler = st.session_state.trace_handler
    trace_handler.reset()
    graph = _get_graph()
    current_state = st.session_state.chat_state
    logger.info(
        "ui turn start query='%s' history_items=%s",
        truncate_text(user_query, 120),
        len(current_state.get("history", [])),
    )
    try:
        next_state = graph.invoke(
            {
                **current_state,
                "user_query": user_query,
                "final_response": "",
                "retrieved_context": [],
            }
        )
    except Exception as exc:
        logger.exception("ui turn failed: %s", exc)
        st.session_state.turn_logs = trace_handler.snapshot()
        raise
    st.session_state.chat_state = next_state
    logger.info(
        "ui turn completed final_response='%s'",
        truncate_text(next_state.get("final_response", ""), 140),
    )
    turn_logs = trace_handler.snapshot()
    st.session_state.turn_logs = turn_logs
    return {
        "content": next_state.get("final_response", "").strip()
        or "I wasn't able to generate a response.",
        "retrieval_query": str(next_state.get("retrieval_query", "")).strip(),
        "retrieved_context": list(next_state.get("retrieved_context", [])),
        "turn_logs": turn_logs,
    }


def main() -> None:
    st.set_page_config(
        page_title="Customer Care AI Demo",
        page_icon="💬",
        layout="centered",
    )
    _ensure_session_state()

    st.title("Customer Care AI Chat")
    st.caption("Demo UI for testing KB answers, bookings, and escalation flows.")
    _render_sidebar()

    for message in st.session_state.messages:
        _render_message(message)

    user_query = st.chat_input("Ask about services, appointments, or support")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_message = _run_turn(user_query)
            except Exception:
                assistant_message = {
                    "content": (
                        "The backend hit an error while processing this turn. "
                        "Check the Backend Trace panel for details."
                    ),
                    "retrieval_query": "",
                    "retrieved_context": [],
                    "turn_logs": st.session_state.turn_logs,
                }
        st.markdown(str(assistant_message["content"]))

        retrieval_query = str(assistant_message.get("retrieval_query", "")).strip()
        if retrieval_query:
            with st.expander("Vector DB Query", expanded=False):
                st.code(retrieval_query, language="text")

        retrieved_context = assistant_message.get("retrieved_context", [])
        if isinstance(retrieved_context, list) and retrieved_context:
            with st.expander(
                f"Retrieved Chunks ({len(retrieved_context)})",
                expanded=False,
            ):
                for index, item in enumerate(retrieved_context, start=1):
                    st.caption(f"Chunk {index}")
                    st.code(str(item), language="text")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(assistant_message["content"]),
            "retrieval_query": str(assistant_message.get("retrieval_query", "")),
            "retrieved_context": list(assistant_message.get("retrieved_context", [])),
            "turn_logs": list(assistant_message.get("turn_logs", [])),
        }
    )


if __name__ == "__main__":
    main()
