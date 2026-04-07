from __future__ import annotations

from app.agents.contracts import StateAgent
from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update
from app.services.contracts import EscalationService

logger = get_logger("agents.escalation")


class HumanEscalationAgent(StateAgent):
    def __init__(self, escalation_service: EscalationService) -> None:
        self._escalation_service = escalation_service

    def execute(self, state: ChatState) -> ChatState:
        logger.info("escalation agent received state: %s", summarize_state(state))
        update = {
            "intent": "human_escalation",
            "handoff_pending": True,
            "active_action": None,
            "appointment_slots": {},
            "missing_slots": [],
            "available_dates": [],
            "date_confirmed": False,
            "available_slots": [],
            "time_confirmed": False,
            "awaiting_confirmation": False,
            "final_response": self._escalation_service.build_response(state),
        }
        logger.info("escalation agent produced update: %s", summarize_update(update))
        return update
