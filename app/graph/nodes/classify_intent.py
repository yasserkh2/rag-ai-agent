from app.observability import get_logger, summarize_state, summarize_update, truncate_text
from app.graph.state import ChatState
from app.services.contracts import IntentClassifier
from app.services.intent import KeywordIntentClassifier

logger = get_logger("graph.nodes.classify_intent")


class ClassifyIntentNode:
    def __init__(self, intent_classifier: IntentClassifier) -> None:
        self._intent_classifier = intent_classifier

    def __call__(self, state: ChatState) -> ChatState:
        logger.info(
            "classify_intent starting: query='%s' state=%s",
            truncate_text(state.get("user_query", ""), 100),
            summarize_state(state),
        )
        decision = self._intent_classifier.classify(state)
        update = decision.as_state_update()
        logger.info("classify_intent completed: %s", summarize_update(update))
        return update


_default_node = ClassifyIntentNode(KeywordIntentClassifier())


def classify_intent(state: ChatState) -> ChatState:
    return _default_node(state)
