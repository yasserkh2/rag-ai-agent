from __future__ import annotations

import unittest

from app.graph.router import ActiveFlowRouter, PostTurnRouter, ServiceResultRouter
from app.services.router import DefaultIntentRouter


class ActiveFlowRouterTests(unittest.TestCase):
    def test_routes_handoff_pending_turn_directly_to_human_escalation(self) -> None:
        router = ActiveFlowRouter()

        route = router(
            {"handoff_pending": True, "active_action": "appointment_scheduling"}
        )

        self.assertEqual(route, "human_escalation")

    def test_routes_active_appointment_flow_directly_to_action_request(self) -> None:
        router = ActiveFlowRouter()

        route = router({"active_action": "appointment_scheduling"})

        self.assertEqual(route, "action_request")

    def test_routes_to_classify_intent_when_no_active_action_exists(self) -> None:
        router = ActiveFlowRouter()

        route = router({"active_action": None})

        self.assertEqual(route, "classify_intent")


class ServiceResultRouterTests(unittest.TestCase):
    def test_routes_service_result_to_evaluate_escalation_by_default(self) -> None:
        router = ServiceResultRouter()

        route = router({"handoff_pending": False})

        self.assertEqual(route, "evaluate_escalation")


class PostTurnRouterTests(unittest.TestCase):
    def test_routes_evaluator_result_to_human_when_handoff_is_pending(self) -> None:
        router = PostTurnRouter()

        route = router({"handoff_pending": True})

        self.assertEqual(route, "human_escalation")

    def test_routes_evaluator_result_to_response_when_no_handoff_is_pending(self) -> None:
        router = PostTurnRouter()

        route = router({"handoff_pending": False})

        self.assertEqual(route, "response")


class DefaultIntentRouterTests(unittest.TestCase):
    def test_routes_general_conversation_intent(self) -> None:
        router = DefaultIntentRouter()

        route = router.route({"intent": "general_conversation", "handoff_pending": False})

        self.assertEqual(route, "general_conversation")

    def test_falls_back_to_general_conversation_for_unknown_intent(self) -> None:
        router = DefaultIntentRouter()

        route = router.route({"intent": "unknown", "handoff_pending": False})

        self.assertEqual(route, "general_conversation")


if __name__ == "__main__":
    unittest.main()
