from app.agents import GeneralConversationAgent
from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update
from app.services.responses import GeneralConversationService

logger = get_logger("graph.nodes.general_conversation")


class GeneralConversationNode:
    def __init__(self, agent: GeneralConversationAgent) -> None:
        self._agent = agent

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("general_conversation starting: %s", summarize_state(state))
        update = self._agent.execute(state)
        logger.info("general_conversation completed: %s", summarize_update(update))
        return update


_default_node = GeneralConversationNode(
    GeneralConversationAgent(GeneralConversationService())
)


def general_conversation(state: ChatState) -> ChatState:
    return _default_node(state)
