from __future__ import annotations

import unittest

from app.services.intent import KeywordIntentClassifier, LlmIntentClassifier
from app.services.models import IntentDecision


class StubIntentDecisionGenerator:
    def __init__(self, decision: IntentDecision) -> None:
        self._decision = decision
        self.calls: list[dict[str, object]] = []

    def classify_intent(
        self,
        user_query: str,
        conversation_history: list[str],
        active_action: str | None,
        failure_count: int,
    ) -> IntentDecision:
        self.calls.append(
            {
                "user_query": user_query,
                "conversation_history": list(conversation_history),
                "active_action": active_action,
                "failure_count": failure_count,
            }
        )
        return self._decision


class FailingIntentDecisionGenerator:
    def classify_intent(
        self,
        user_query: str,
        conversation_history: list[str],
        active_action: str | None,
        failure_count: int,
    ) -> IntentDecision:
        raise RuntimeError("intent classifier offline")


class KeywordIntentClassifierTests(unittest.TestCase):
    def test_active_appointment_flow_stays_routed_to_action_request(self) -> None:
        classifier = KeywordIntentClassifier()

        result = classifier.classify(
            {
                "user_query": "what are the available services",
                "active_action": "appointment_scheduling",
            }
        )

        self.assertEqual(result.intent, "action_request")
        self.assertGreaterEqual(result.confidence, 0.95)

    def test_routes_to_human_escalation_for_escalation_request_with_typo(self) -> None:
        classifier = KeywordIntentClassifier()

        result = classifier.classify(
            {
                "user_query": "i need to escilate",
            }
        )

        self.assertEqual(result.intent, "human_escalation")
        self.assertIn("human", result.escalation_reason or "")

    def test_routes_to_human_escalation_for_explicit_handoff_phrase(self) -> None:
        classifier = KeywordIntentClassifier()

        result = classifier.classify(
            {
                "user_query": "please connect me to support",
            }
        )

        self.assertEqual(result.intent, "human_escalation")

    def test_routes_greeting_to_general_conversation(self) -> None:
        classifier = KeywordIntentClassifier()

        result = classifier.classify(
            {
                "user_query": "hi",
            }
        )

        self.assertEqual(result.intent, "general_conversation")

    def test_short_follow_up_uses_history_to_continue_action_request(self) -> None:
        classifier = KeywordIntentClassifier()

        result = classifier.classify(
            {
                "user_query": "Thursday",
                "history": [
                    "user: I want to book a meeting",
                    "assistant: I found available dates: Next Thursday, Next Friday, Next Monday. Which date would you like?",
                ],
                "active_action": None,
            }
        )

        self.assertEqual(result.intent, "action_request")


class LlmIntentClassifierTests(unittest.TestCase):
    def test_uses_llm_decision_when_generator_succeeds(self) -> None:
        generator = StubIntentDecisionGenerator(
            IntentDecision(
                intent="human_escalation",
                confidence=0.97,
                frustration_flag=True,
                escalation_reason="User asked for a supervisor.",
            )
        )
        classifier = LlmIntentClassifier(decision_generator=generator)

        result = classifier.classify(
            {
                "user_query": "I want a supervisor",
                "history": ["user: hi", "assistant: hello"],
                "active_action": "appointment_scheduling",
                "failure_count": 2,
            }
        )

        self.assertEqual(result.intent, "human_escalation")
        self.assertTrue(result.frustration_flag)
        self.assertEqual(result.escalation_reason, "User asked for a supervisor.")
        self.assertEqual(generator.calls[0]["failure_count"], 2)
        self.assertEqual(
            generator.calls[0]["conversation_history"],
            ["user: hi", "assistant: hello"],
        )

    def test_uses_llm_general_conversation_decision_when_generator_succeeds(self) -> None:
        generator = StubIntentDecisionGenerator(
            IntentDecision(
                intent="general_conversation",
                confidence=0.88,
                frustration_flag=False,
                escalation_reason=None,
            )
        )
        classifier = LlmIntentClassifier(decision_generator=generator)

        result = classifier.classify(
            {
                "user_query": "hi there",
                "history": ["user: hi"],
                "active_action": None,
                "failure_count": 0,
            }
        )

        self.assertEqual(result.intent, "general_conversation")

    def test_falls_back_to_keyword_classifier_when_generator_fails(self) -> None:
        classifier = LlmIntentClassifier(
            decision_generator=FailingIntentDecisionGenerator()
        )

        result = classifier.classify(
            {
                "user_query": "i need to escilate",
            }
        )

        self.assertEqual(result.intent, "human_escalation")


if __name__ == "__main__":
    unittest.main()
