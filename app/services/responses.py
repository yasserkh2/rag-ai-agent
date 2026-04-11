from __future__ import annotations

from app.graph.state import ChatState
from app.llm.contracts import EscalationReplyGenerator
from app.observability import get_logger

logger = get_logger("services.responses")


class HumanEscalationService:
    def __init__(
        self,
        escalation_reply_generator: EscalationReplyGenerator | None = None,
    ) -> None:
        self._escalation_reply_generator = escalation_reply_generator

    def build_response(self, state: ChatState) -> str:
        fallback_message = self._build_template_response(state)
        generator = self._escalation_reply_generator
        if generator is None:
            return fallback_message

        reason = str(state.get("escalation_reason") or "This request needs human support.").strip()
        name = str(state.get("escalation_contact_name") or "").strip() or None
        email = str(state.get("escalation_contact_email") or "").strip() or None
        phone = str(state.get("escalation_contact_phone") or "").strip() or None
        escalation_case_id = str(state.get("escalation_case_id") or "").strip() or None
        user_query = str(state.get("user_query") or "").strip()
        history = list(state.get("history", []))
        requires_contact = not bool(email or phone)
        try:
            llm_response = generator.generate_reply(
                user_query=user_query,
                escalation_reason=reason,
                conversation_history=history,
                escalation_case_id=escalation_case_id,
                contact_name=name,
                contact_email=email,
                contact_phone=phone,
                requires_contact=requires_contact,
            )
            if llm_response.strip():
                return llm_response.strip()
        except Exception as exc:
            logger.warning("escalation reply generation failed, using template fallback: %s", exc)

        return fallback_message

    @staticmethod
    def _build_template_response(state: ChatState) -> str:
        reason = state.get("escalation_reason") or "This request needs human support."
        name = str(state.get("escalation_contact_name") or "").strip()
        email = str(state.get("escalation_contact_email") or "").strip()
        phone = str(state.get("escalation_contact_phone") or "").strip()
        escalation_case_id = str(state.get("escalation_case_id") or "").strip()
        base_message = (
            "I need to transfer this conversation to a human agent. "
            "A human agent will follow up with you. "
            f"Reason: {reason}"
        )
        contact_channels = [channel for channel in (email, phone) if channel]
        if contact_channels:
            if len(contact_channels) == 2:
                contact_text = f"{contact_channels[0]} or {contact_channels[1]}"
            else:
                contact_text = contact_channels[0]
            thanks_prefix = f"Thanks, {name}. " if name else "Thank you. "
            case_suffix = (
                f"Your escalation reference is {escalation_case_id}. "
                if escalation_case_id
                else ""
            )
            return (
                f"{base_message} {thanks_prefix}{case_suffix}"
                f"I've shared this with our human team. They'll reach out at {contact_text} shortly."
            )

        return (
            f"{base_message} "
            "Please share your name and either a valid phone number or email, "
            "and our human team will follow up with you."
        )


class GeneralConversationService:
    def build_response(self, state: ChatState) -> str:
        query = str(state.get("user_query", "")).strip().lower()

        if query in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            return (
                "Hello! I can help explain our services, book a consultation, "
                "or connect you with a human agent. Share what you need."
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
            "connecting you with a human agent. Tell me what you need."
        )
