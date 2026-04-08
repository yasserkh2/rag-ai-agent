from __future__ import annotations

from app.graph.state import ChatState
from app.services.models import KnowledgeBaseAnswer


class PlaceholderKnowledgeBaseService:
    def answer(self, state: ChatState) -> KnowledgeBaseAnswer:
        query = state.get("user_query", "")
        return KnowledgeBaseAnswer(
            final_response=(
                "This is a placeholder KB answer path. "
                f"Later we will retrieve grounded context for: '{query}'."
            )
        )


class AppointmentRequestService:
    def build_response(self, state: ChatState) -> str:
        return (
            "I can help with an appointment request. "
            "Please share the service you need, plus your preferred date and time."
        )


class HumanEscalationService:
    def build_response(self, state: ChatState) -> str:
        reason = state.get("escalation_reason") or "This request needs human support."
        return (
            "I need to transfer this conversation to a human agent. "
            "A human agent will follow up with you. "
            f"Reason: {reason}"
        )


class GeneralConversationService:
    def build_response(self, state: ChatState) -> str:
        query = str(state.get("user_query", "")).strip().lower()

        if query in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            return (
                "Hello! I can help explain our services, book a consultation, "
                "or connect you with a human agent. What would you like to do?"
            )

        if query in {"thanks", "thank you", "ok", "okay"}:
            return (
                "You're welcome. I can also answer service questions, help you "
                "schedule a consultation, or connect you with a human agent."
            )

        if any(
            phrase in query
            for phrase in {
                "what can you do",
                "help me",
                "can you help",
                "how can you help",
                "what do you do",
            }
        ):
            return (
                "I can help with service information, consultation booking, and "
                "human handoff when needed. You can ask about a service or ask me "
                "to schedule a meeting."
            )

        if query in {"bye", "goodbye", "see you"}:
            return (
                "You're welcome to come back anytime. I can help with service "
                "questions, consultation booking, or human support."
            )

        return (
            "I can help with service information, scheduling a consultation, or "
            "connecting you with a human agent. Tell me what you'd like to do."
        )
