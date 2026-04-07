from __future__ import annotations

from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update

logger = get_logger("services.escalation")


class PostTurnEscalationEvaluator:
    def __init__(self, repeated_failure_threshold: int = 3) -> None:
        self._repeated_failure_threshold = repeated_failure_threshold

    def evaluate(self, state: ChatState) -> ChatState:
        logger.info("post-turn escalation evaluating: %s", summarize_state(state))
        if state.get("handoff_pending"):
            update = {
                "handoff_pending": True,
                "intent": "human_escalation",
            }
            logger.info("post-turn escalation preserved handoff: %s", summarize_update(update))
            return update

        failure_count = int(state.get("failure_count", 0))
        turn_outcome = state.get("turn_outcome")
        escalation_reason = state.get("escalation_reason")
        frustration_flag = bool(state.get("frustration_flag"))

        if turn_outcome == "unresolved":
            failure_count += 1
        elif turn_outcome in {"resolved", "needs_input"}:
            failure_count = 0

        if frustration_flag and not escalation_reason:
            escalation_reason = (
                "The conversation appears frustrated and needs human support."
            )

        if escalation_reason:
            update = {
                "failure_count": failure_count,
                "handoff_pending": True,
                "intent": "human_escalation",
                "escalation_reason": escalation_reason,
            }
            logger.info("post-turn escalation triggered by reason: %s", summarize_update(update))
            return update

        if turn_outcome == "unresolved" and failure_count >= self._repeated_failure_threshold:
            update = {
                "failure_count": failure_count,
                "handoff_pending": True,
                "intent": "human_escalation",
                "escalation_reason": self._build_repeated_failure_reason(state),
            }
            logger.info("post-turn escalation triggered by repeated failure: %s", summarize_update(update))
            return update

        update = {
            "failure_count": failure_count,
        }
        logger.info("post-turn escalation completed without handoff: %s", summarize_update(update))
        return update

    @staticmethod
    def _build_repeated_failure_reason(state: ChatState) -> str:
        turn_failure_reason = state.get("turn_failure_reason")
        if turn_failure_reason:
            return (
                "I need to transfer this conversation to a human agent because "
                f"the issue remains unresolved after repeated attempts "
                f"({turn_failure_reason})."
            )
        return (
            "I need to transfer this conversation to a human agent because the "
            "issue remains unresolved after repeated attempts."
        )
