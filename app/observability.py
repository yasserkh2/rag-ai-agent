from __future__ import annotations

import logging
from collections.abc import Mapping
from threading import Lock
from typing import Any

from app.graph.state import ChatState

LOGGER_NAME = "customer_care_ai"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def summarize_state(state: ChatState) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "active_action": state.get("active_action"),
        "handoff_pending": state.get("handoff_pending"),
        "failure_count": state.get("failure_count"),
        "turn_outcome": state.get("turn_outcome"),
        "awaiting_confirmation": state.get("awaiting_confirmation"),
        "missing_slots": list(state.get("missing_slots", [])),
        "retrieved_context_count": len(state.get("retrieved_context", [])),
    }


def summarize_update(update: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "intent",
        "confidence",
        "active_action",
        "handoff_pending",
        "failure_count",
        "turn_outcome",
        "turn_failure_reason",
        "escalation_reason",
        "awaiting_confirmation",
        "booking_error",
        "booking_confirmation_id",
    ):
        if key in update:
            summary[key] = update[key]

    if "appointment_slots" in update:
        slots = update["appointment_slots"]
        summary["appointment_slots"] = slots if isinstance(slots, dict) else update["appointment_slots"]
    if "available_dates" in update:
        summary["available_dates"] = list(update["available_dates"])
    if "available_slots" in update:
        summary["available_slots"] = list(update["available_slots"])
    if "retrieved_context" in update:
        summary["retrieved_context_count"] = len(update["retrieved_context"])
    if "final_response" in update:
        summary["final_response_preview"] = truncate_text(str(update["final_response"]), 140)
    return summary


def truncate_text(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


class InMemoryLogHandler(logging.Handler):
    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._records: list[str] = []
        self._lock = Lock()
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        rendered = self.format(record)
        with self._lock:
            self._records.append(rendered)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._records)
