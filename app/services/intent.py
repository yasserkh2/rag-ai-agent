from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.graph.state import ChatState
from app.llm.contracts import IntentDecisionGenerator
from app.llm.intent_factory import IntentDecisionGeneratorFactory
from app.observability import get_logger, summarize_state, truncate_text
from app.services.models import IntentDecision

logger = get_logger("services.intent")


@dataclass(frozen=True, slots=True)
class IntentKeywordCatalog:
    action_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"appointment", "book", "booking", "schedule", "meeting"}
        )
    )
    escalation_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "human",
                "agent",
                "manager",
                "complaint",
                "representative",
                "supervisor",
                "transfer",
                "handoff",
                "escalat",
                "escilat",
            }
        )
    )
    frustration_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"angry", "frustrated", "upset", "annoyed", "terrible"}
        )
    )
    general_conversation_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "hello",
                "hi",
                "hey",
                "thanks",
                "thank you",
                "ok",
                "okay",
                "bye",
                "goodbye",
                "help",
            }
        )
    )


class KeywordIntentClassifier:
    def __init__(self, keyword_catalog: IntentKeywordCatalog | None = None) -> None:
        self._keyword_catalog = keyword_catalog or IntentKeywordCatalog()

    def classify(self, state: ChatState) -> IntentDecision:
        query = state.get("user_query", "")
        normalized_query = query.lower()
        active_action = state.get("active_action")
        history = [str(item).strip() for item in state.get("history", []) if str(item).strip()]
        frustration_flag = self._contains_any(
            normalized_query, self._keyword_catalog.frustration_keywords
        )
        logger.info(
            "keyword classifier evaluating query='%s' active_action=%s history_items=%s",
            truncate_text(query, 100),
            active_action,
            len(history),
        )

        if frustration_flag or self._is_explicit_escalation_request(normalized_query):
            decision = IntentDecision(
                intent="human_escalation",
                confidence=0.9,
                frustration_flag=frustration_flag,
                escalation_reason=(
                    "User requested help from a human or showed frustration."
                ),
            )
            logger.info("keyword classifier chose human_escalation")
            return decision

        if active_action == "appointment_scheduling":
            decision = IntentDecision(
                intent="action_request",
                confidence=0.95,
                frustration_flag=frustration_flag,
            )
            logger.info("keyword classifier stayed in action_request")
            return decision

        if self._should_continue_action_from_history(
            normalized_query=normalized_query,
            history=history,
        ):
            decision = IntentDecision(
                intent="action_request",
                confidence=0.82,
                frustration_flag=frustration_flag,
            )
            logger.info("keyword classifier continued action_request from history")
            return decision

        if self._contains_any(normalized_query, self._keyword_catalog.action_keywords):
            decision = IntentDecision(
                intent="action_request",
                confidence=0.85,
                frustration_flag=frustration_flag,
            )
            logger.info("keyword classifier chose action_request")
            return decision

        if self._is_general_conversation_turn(normalized_query):
            decision = IntentDecision(
                intent="general_conversation",
                confidence=0.8,
                frustration_flag=frustration_flag,
            )
            logger.info("keyword classifier chose general_conversation")
            return decision

        decision = IntentDecision(
            intent="kb_query",
            confidence=0.65,
            frustration_flag=frustration_flag,
        )
        logger.info("keyword classifier chose kb_query")
        return decision

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_explicit_escalation_request(self, text: str) -> bool:
        if self._contains_any(text, self._keyword_catalog.escalation_keywords):
            return True

        escalation_phrases = (
            "talk to a human",
            "talk to an agent",
            "real person",
            "need a human",
            "connect me to support",
            "speak to someone",
        )
        return any(phrase in text for phrase in escalation_phrases)

    def _is_general_conversation_turn(self, text: str) -> bool:
        normalized = text.strip()
        if normalized in self._keyword_catalog.general_conversation_keywords:
            return True

        general_phrases = (
            "what can you do",
            "can you help",
            "help me",
            "how can you help",
            "what do you do",
        )
        return any(phrase in normalized for phrase in general_phrases)

    def _should_continue_action_from_history(
        self,
        *,
        normalized_query: str,
        history: list[str],
    ) -> bool:
        if not normalized_query or not history:
            return False

        if not self._looks_like_short_follow_up(normalized_query):
            return False

        recent_history = history[-4:]
        recent_assistant_messages = [
            message.removeprefix("assistant:").strip().lower()
            for message in recent_history
            if message.lower().startswith("assistant:")
        ]

        action_cues = (
            "which service would you like to book",
            "which service would you like",
            "what date would you like",
            "which date would you like",
            "available dates",
            "available times",
            "which time would you like",
            "what time would work",
            "what name should i use",
            "what email address should i use",
            "please confirm your appointment",
            "please confirm the appointment",
        )
        return any(
            cue in message
            for message in recent_assistant_messages
            for cue in action_cues
        )

    @staticmethod
    def _looks_like_short_follow_up(normalized_query: str) -> bool:
        short_follow_up_phrases = {
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "sure",
            "tomorrow",
            "today",
            "thursday",
            "friday",
            "monday",
            "tuesday",
            "wednesday",
            "saturday",
            "sunday",
            "morning",
            "afternoon",
            "evening",
            "the first one",
            "the second one",
            "the third one",
            "first one",
            "second one",
            "third one",
            "book it",
            "confirm it",
        }
        if normalized_query in short_follow_up_phrases:
            return True

        if len(normalized_query.split()) <= 4:
            if any(char.isdigit() for char in normalized_query):
                return True
            if "@" in normalized_query:
                return True

        return False


class LlmIntentClassifier:
    def __init__(
        self,
        decision_generator: IntentDecisionGenerator | None = None,
        fallback_classifier: KeywordIntentClassifier | None = None,
    ) -> None:
        self._fallback_classifier = fallback_classifier or KeywordIntentClassifier()
        if decision_generator is not None:
            self._decision_generator = decision_generator
        else:
            self._decision_generator = self._build_generator()

    def classify(self, state: ChatState) -> IntentDecision:
        if self._decision_generator is None:
            logger.info("llm intent classifier unavailable, using keyword fallback")
            return self._fallback_classifier.classify(state)

        try:
            logger.info("llm intent classifier evaluating state=%s", summarize_state(state))
            decision = self._decision_generator.classify_intent(
                user_query=state.get("user_query", ""),
                conversation_history=list(state.get("history", [])),
                active_action=state.get("active_action"),
                failure_count=int(state.get("failure_count", 0)),
            )
            logger.info(
                "llm intent classifier chose intent=%s confidence=%s frustration=%s",
                decision.intent,
                decision.confidence,
                decision.frustration_flag,
            )
            return decision
        except Exception as exc:
            logger.exception("llm intent classifier failed, using keyword fallback: %s", exc)
            return self._fallback_classifier.classify(state)

    @staticmethod
    def _build_generator() -> IntentDecisionGenerator | None:
        try:
            return IntentDecisionGeneratorFactory().build()
        except Exception:
            return None
