from app.agents import KnowledgeBaseAgent
from app.graph.state import ChatState
from app.observability import get_logger, summarize_state, summarize_update
from app.services.knowledge_base import RetrievalKnowledgeBaseService

logger = get_logger("graph.nodes.kb_answer")


class KnowledgeBaseAnswerNode:
    def __init__(self, agent: KnowledgeBaseAgent) -> None:
        self._agent = agent

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("kb_answer starting: %s", summarize_state(state))
        update = self._agent.execute(state)
        logger.info("kb_answer completed: %s", summarize_update(update))
        return update


_default_node = KnowledgeBaseAnswerNode(
    KnowledgeBaseAgent(RetrievalKnowledgeBaseService())
)


def kb_answer(state: ChatState) -> ChatState:
    return _default_node(state)
