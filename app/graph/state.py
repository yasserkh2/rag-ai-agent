from typing import Any, Literal, TypedDict


Intent = Literal[
    "kb_query",
    "action_request",
    "human_escalation",
    "general_conversation",
    "unknown",
]
TurnOutcome = Literal["resolved", "needs_input", "unresolved"]


class ChatState(TypedDict, total=False):
    user_query: str
    retrieval_query: str
    intent: Intent
    confidence: float
    entities: dict[str, Any]
    history: list[str]
    handoff_pending: bool
    failure_count: int
    turn_outcome: TurnOutcome
    turn_failure_reason: str | None
    frustration_flag: bool
    escalation_reason: str | None
    retrieved_context: list[str]
    active_action: str | None
    appointment_slots: dict[str, str]
    missing_slots: list[str]
    available_dates: list[str]
    suggested_date: str | None
    date_confirmed: bool
    available_slots: list[str]
    suggested_time: str | None
    time_confirmed: bool
    awaiting_confirmation: bool
    booking_confirmation_id: str | None
    booking_result: dict[str, Any] | None
    booking_error: str | None
    final_response: str


def create_initial_state(user_query: str) -> ChatState:
    return {
        "user_query": user_query,
        "retrieval_query": "",
        "intent": "unknown",
        "confidence": 0.0,
        "entities": {},
        "history": [],
        "handoff_pending": False,
        "failure_count": 0,
        "turn_outcome": "resolved",
        "turn_failure_reason": None,
        "frustration_flag": False,
        "escalation_reason": None,
        "retrieved_context": [],
        "active_action": None,
        "appointment_slots": {},
        "missing_slots": [],
        "available_dates": [],
        "suggested_date": None,
        "date_confirmed": False,
        "available_slots": [],
        "suggested_time": None,
        "time_confirmed": False,
        "awaiting_confirmation": False,
        "booking_confirmation_id": None,
        "booking_result": None,
        "booking_error": None,
        "final_response": "",
    }
