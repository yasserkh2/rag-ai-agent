from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update
from app.services.contracts import EscalationEvaluator
from app.services.escalation import PostTurnEscalationEvaluator

logger = get_logger("graph.nodes.evaluate_escalation")


class EvaluateEscalationNode:
    def __init__(self, evaluator: EscalationEvaluator) -> None:
        self._evaluator = evaluator

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("evaluate_escalation starting: %s", summarize_state(state))
        update = self._evaluator.evaluate(state)
        logger.info("evaluate_escalation completed: %s", summarize_update(update))
        return update


_default_node = EvaluateEscalationNode(PostTurnEscalationEvaluator())


def evaluate_escalation(state: ChatState) -> ChatState:
    return _default_node(state)
