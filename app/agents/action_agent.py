from __future__ import annotations

from app.agents.contracts import StateAgent
from app.graph.state import ChatState
from app.observability import get_logger, summarize_update
from app.services.contracts import ActionRequestService

logger = get_logger("agents.action")


class ActionRequestAgent(StateAgent):
    def __init__(self, action_request_service: ActionRequestService) -> None:
        self._action_request_service = action_request_service

    def execute(self, state: ChatState) -> ChatState:
        update = self._action_request_service.handle_turn(state)
        logger.info("action agent produced update: %s", summarize_update(update))
        return update
