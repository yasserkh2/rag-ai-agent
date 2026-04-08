from __future__ import annotations

from app.graph.state import ChatState, Intent


class DefaultIntentRouter:
    def __init__(self, fallback_intent: Intent = "general_conversation") -> None:
        self._fallback_intent = fallback_intent
        self._valid_routes = frozenset(
            {"kb_query", "action_request", "human_escalation", "general_conversation"}
        )

    def route(self, state: ChatState) -> str:
        if state.get("handoff_pending"):
            return "human_escalation"

        if state.get("frustration_flag"):
            return "human_escalation"

        intent = state.get("intent", self._fallback_intent)
        if intent in self._valid_routes:
            return intent

        return self._fallback_intent
