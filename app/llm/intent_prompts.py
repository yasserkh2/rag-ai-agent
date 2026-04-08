from __future__ import annotations

import json


DEFAULT_INTENT_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify the latest user turn for a customer-care assistant into one of three intents. "
    "Your job is to choose the route that creates the best user experience while staying safe and accurate. "
    "Return JSON only.\n"
    "\n"
    "Valid intents:\n"
    '- "kb_query" for informational questions, service explanations, policy questions, pricing or scope questions, and general help requests where the assistant should answer from knowledge.\n'
    '- "action_request" for turns where the user wants to start, continue, confirm, modify, or complete a supported workflow such as booking a consultation or providing booking details.\n'
    '- "human_escalation" when the user explicitly wants a person, asks to escalate, is clearly frustrated, reports repeated failure, or the conversation should be handed to a human for better support.\n'
    '- "general_conversation" for greetings, thanks, small talk, vague conversational turns, or broad capability/help turns where the assistant should reply conversationally without retrieval, workflow execution, or handoff.\n'
    "\n"
    "Routing principles:\n"
    "- Classify the turn using the conversation thread, not the latest message in isolation.\n"
    "- If the latest message is short or ambiguous, infer intent from recent assistant questions, the active action, and the current conversational direction.\n"
    "- Prefer action_request when the user is trying to complete a task, even if the wording is short, informal, or incomplete.\n"
    "- Prefer kb_query when the user mainly wants information, options, explanations, or service details before deciding on an action.\n"
    "- Prefer general_conversation for greetings, thanks, short social turns, or general capability questions where the assistant should guide the user naturally.\n"
    "- Prefer human_escalation only when there is a real handoff signal, not merely because the user sounds brief or uncertain.\n"
    "- If an action flow is already active, keep the intent as action_request unless the user clearly asks for a human or clearly abandons the workflow in favor of escalation.\n"
    "- Messages like dates, times, names, emails, yes/no confirmations, and option selections should usually stay in action_request when an action flow is active.\n"
    "- Messages like 'tomorrow', 'thursday', '10:30', 'the first one', 'yes', or an email address should usually continue the current workflow when the recent conversation shows the assistant was collecting booking details.\n"
    "- Questions like 'what services do you offer', 'what does this include', or 'how does it work' should usually be kb_query unless they are clearly part of an active booking flow.\n"
    "- Treat words such as escalate, escalation, escilate, transfer, handoff, supervisor, human, representative, real person, or talk to support as strong escalation signals when used as a request for a person.\n"
    "- Mild dissatisfaction alone does not always mean human_escalation, but strong frustration or repeated failure should.\n"
    "\n"
    "Frustration guidance:\n"
    "- Set frustration_flag to true when the user sounds angry, upset, frustrated, disappointed, or says the bot is not helping.\n"
    "- frustration_flag may be true even if intent remains kb_query or action_request.\n"
    "- Set escalation_reason only when intent is human_escalation.\n"
    "\n"
    "Confidence guidance:\n"
    "- Confidence must be a number between 0 and 1.\n"
    "- Use higher confidence when the signal is explicit and lower confidence when the turn is ambiguous.\n"
    "\n"
    "Examples:\n"
    '- "What does Website Services include?" -> kb_query\n'
    '- "I want to book a meeting" -> action_request\n'
    '- "Thursday works" during booking -> action_request\n'
    '- "yes, book it" during booking -> action_request\n'
    '- "I need a human agent" -> human_escalation\n'
    '- "this is the third time, you are not helping" -> human_escalation with frustration_flag true\n'
    '- "hi" -> general_conversation\n'
    '- "thanks" -> general_conversation\n'
    '- "what can you do?" -> general_conversation\n'
    "\n"
    "Return valid JSON with exactly these keys and no extras:\n"
    '{"intent": "kb_query", "confidence": 0.0, "frustration_flag": false, "escalation_reason": null}\n'
)


def build_intent_classifier_prompt(
    user_query: str,
    conversation_history: list[str],
    active_action: str | None,
    failure_count: int,
) -> str:
    history_block = "\n".join(conversation_history[-6:]) or "[no prior conversation]"
    return (
        "Classify the latest customer turn for routing.\n\n"
        "Recent conversation:\n"
        f"{history_block}\n\n"
        "Active action:\n"
        f"{active_action or '[none]'}\n\n"
        "Current failure count:\n"
        f"{failure_count}\n\n"
        "Latest user message:\n"
        f"{user_query}\n\n"
        "Think about the best next route for the user: knowledge answer, workflow continuation, conversational guidance, or human handoff.\n"
        "Use the recent conversation as primary context when the latest message is short, referential, or ambiguous.\n\n"
        "Return valid JSON with exactly these keys:\n"
        '{"intent": "kb_query", "confidence": 0.0, "frustration_flag": false, "escalation_reason": null}'
    )


def build_history_block(conversation_history: list[str]) -> str:
    return "\n".join(message.strip() for message in conversation_history[-6:] if message.strip()) or "[no prior conversation]"


def parse_intent_decision_payload(payload: dict[str, object]) -> dict[str, object]:
    intent = str(payload.get("intent", "")).strip()
    if intent not in {"kb_query", "action_request", "human_escalation", "general_conversation"}:
        raise RuntimeError("Intent classification payload contained an invalid intent.")

    confidence_value = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Intent classification payload contained an invalid confidence.") from exc

    confidence = max(0.0, min(1.0, confidence))
    frustration_flag = bool(payload.get("frustration_flag", False))
    escalation_reason = payload.get("escalation_reason")
    if escalation_reason is not None:
        escalation_reason = str(escalation_reason).strip() or None

    return {
        "intent": intent,
        "confidence": confidence,
        "frustration_flag": frustration_flag,
        "escalation_reason": escalation_reason,
    }


def build_intent_classifier_debug_payload(
    user_query: str,
    conversation_history: list[str],
    active_action: str | None,
    failure_count: int,
) -> str:
    return json.dumps(
        {
            "user_query": user_query,
            "conversation_history": conversation_history[-6:],
            "active_action": active_action,
            "failure_count": failure_count,
        },
        sort_keys=True,
    )
