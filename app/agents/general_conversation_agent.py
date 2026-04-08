from __future__ import annotations

from app.agents.contracts import StateAgent
from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update
from app.services.contracts import GeneralConversationService

logger = get_logger("agents.general_conversation")


class GeneralConversationAgent(StateAgent):
    def __init__(self, general_conversation_service: GeneralConversationService) -> None:
        self._general_conversation_service = general_conversation_service

    def execute(self, state: ChatState) -> ChatState:
        logger.info("general conversation agent received state: %s", summarize_state(state))
        update = {
            "intent": "general_conversation",
            "final_response": self._general_conversation_service.build_response(state),
            "turn_outcome": "resolved",
            "turn_failure_reason": None,
            "escalation_reason": None,
        }
        logger.info(
            "general conversation agent produced update: %s",
            summarize_update(update),
        )
        return update
