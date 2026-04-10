from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

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


def _extract_trace_sections(turn_logs: list[str]) -> tuple[list[str], list[str]]:
    routing_lines: list[str] = []
    node_lines: list[str] = []

    for line in turn_logs:
        if any(
            marker in line
            for marker in (
                "active_flow route=",
                "intent route=",
                "service_result route=",
                "post_turn route=",
            )
        ):
            routing_lines.append(line)
            continue

        if any(
            marker in line
            for marker in (
                "graph.nodes.ingest_query",
                "graph.nodes.classify_intent",
                "graph.nodes.general_conversation",
                "graph.nodes.kb_answer",
                "graph.nodes.action_request",
                "graph.nodes.evaluate_escalation",
                "graph.nodes.human_escalation",
                "graph.nodes.response",
            )
        ):
            node_lines.append(line)

    return routing_lines, node_lines


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
                    "Hello, I am COB Company's customer care assistant. "
                    "I can help with company and service questions, guide consultation booking, "
                    "or connect you with a human agent. "
                    "What would you like to start with?"
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
                        "Chat reset. I can help with COB Company services, booking, "
                        "or human support. What would you like to do next?"
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
        routing_lines, node_lines = _extract_trace_sections(st.session_state.turn_logs)

        st.subheader("Routing Trace")
        if routing_lines:
            st.code("\n".join(routing_lines), language="text")
        else:
            st.caption("Routing steps will appear here after the first message.")

        st.divider()
        st.subheader("Node Trace")
        if node_lines:
            st.code("\n".join(node_lines), language="text")
        else:
            st.caption("Node execution steps will appear here after the first message.")

        st.divider()
        st.subheader("Full Backend Trace")
        turn_logs = st.session_state.turn_logs
        if turn_logs:
            st.code("\n".join(turn_logs), language="text")
        else:
            st.caption("Logs will appear here after the first message.")


def _render_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        latency_caption = str(message.get("latency_caption", "")).strip()
        if latency_caption:
            st.caption(latency_caption)

        retrieval_query = str(message.get("retrieval_query", "")).strip()
        if retrieval_query:
            with st.expander("Vector DB Query", expanded=False):
                st.code(retrieval_query, language="text")

        turn_logs = message.get("turn_logs", [])
        if isinstance(turn_logs, list) and turn_logs:
            routing_lines, node_lines = _extract_trace_sections(
                [str(item) for item in turn_logs]
            )
            if routing_lines:
                with st.expander("Routing Trace", expanded=False):
                    st.code("\n".join(routing_lines), language="text")
            if node_lines:
                with st.expander("Node Trace", expanded=False):
                    st.code("\n".join(node_lines), language="text")

        retrieved_context = message.get("retrieved_context", [])
        if isinstance(retrieved_context, list) and retrieved_context:
            with st.expander(
                f"Retrieved Chunks ({len(retrieved_context)})",
                expanded=False,
            ):
                for index, item in enumerate(retrieved_context, start=1):
                    st.caption(f"Chunk {index}")
                    st.code(str(item), language="text")


def _stream_response_text(text: str):
    for token in text.split(" "):
        if token:
            yield f"{token} "


def _run_turn(
    user_query: str,
    progress_placeholder=None,
    response_placeholder=None,
) -> dict[str, object]:
    trace_handler = st.session_state.trace_handler
    trace_handler.reset()
    graph = _get_graph()
    current_state = st.session_state.chat_state
    logger.info(
        "ui turn start query='%s' history_items=%s",
        truncate_text(user_query, 120),
        len(current_state.get("history", [])),
    )
    working_state = {
        **current_state,
        "user_query": user_query,
        "final_response": "",
        "retrieved_context": [],
    }
    try:
        invoke_start = perf_counter()
        response_rendered = False
        if progress_placeholder is not None:
            progress_placeholder.caption("Running: ingest_query")
        for update in graph.stream(working_state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                if isinstance(node_update, dict):
                    working_state.update(node_update)
                    if response_placeholder is not None and not response_rendered:
                        final_response = str(
                            node_update.get("final_response", "")
                        ).strip()
                        if final_response:
                            response_placeholder.write_stream(
                                _stream_response_text(final_response)
                            )
                            response_rendered = True
                if progress_placeholder is not None:
                    progress_placeholder.caption(f"Running: {node_name}")
        backend_invoke_ms = (perf_counter() - invoke_start) * 1000
        next_state = working_state
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
        "backend_invoke_ms": backend_invoke_ms,
        "response_rendered": response_rendered,
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
    # Warm up the backend graph once on page load so the first user turn
    # does not pay graph initialization overhead.
    _get_graph()
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
        turn_start = perf_counter()
        progress_placeholder = st.empty()
        response_placeholder = st.empty()
        try:
            assistant_message = _run_turn(
                user_query,
                progress_placeholder=progress_placeholder,
                response_placeholder=response_placeholder,
            )
        except Exception:
            assistant_message = {
                "content": (
                    "The backend hit an error while processing this turn. "
                    "Check the Backend Trace panel for details."
                ),
                "backend_invoke_ms": 0.0,
                "response_rendered": False,
                "retrieval_query": "",
                "retrieved_context": [],
                "turn_logs": st.session_state.turn_logs,
            }
        progress_placeholder.empty()
        ui_turn_ms = (perf_counter() - turn_start) * 1000
        latency_caption = (
            f"Latency: backend={float(assistant_message.get('backend_invoke_ms', 0.0)):.0f}ms, "
            f"ui_total={ui_turn_ms:.0f}ms"
        )
        if not bool(assistant_message.get("response_rendered", False)):
            response_placeholder.write_stream(
                _stream_response_text(str(assistant_message["content"]))
            )
        st.caption(latency_caption)

        retrieval_query = str(assistant_message.get("retrieval_query", "")).strip()
        if retrieval_query:
            with st.expander("Vector DB Query", expanded=False):
                st.code(retrieval_query, language="text")

        turn_logs = assistant_message.get("turn_logs", [])
        if isinstance(turn_logs, list) and turn_logs:
            routing_lines, node_lines = _extract_trace_sections(
                [str(item) for item in turn_logs]
            )
            if routing_lines:
                with st.expander("Routing Trace", expanded=False):
                    st.code("\n".join(routing_lines), language="text")
            if node_lines:
                with st.expander("Node Trace", expanded=False):
                    st.code("\n".join(node_lines), language="text")

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
            "latency_caption": latency_caption,
            "retrieval_query": str(assistant_message.get("retrieval_query", "")),
            "retrieved_context": list(assistant_message.get("retrieved_context", [])),
            "turn_logs": list(assistant_message.get("turn_logs", [])),
        }
    )


if __name__ == "__main__":
    main()
