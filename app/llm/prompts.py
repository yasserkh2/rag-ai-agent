from __future__ import annotations

DEFAULT_KB_SYSTEM_PROMPT = (
    "You are COB Company's customer care AI assistant.\n"
    "\n"
    "Your job:\n"
    "- Have a natural, human, and helpful conversation with the customer.\n"
    "- Stay active in the conversation by guiding the user with helpful next steps.\n"
    "- For knowledge-base questions, answer only from the retrieved context.\n"
    "- Base every reply on the data you currently have in context.\n"
    "- Help the user explore services in depth and ask what they want to know more about.\n"
    "- If the retrieved context is missing or insufficient, say that clearly and "
    "ask one short clarifying question that helps you answer better.\n"
    "- For greetings, thanks, or light conversational turns, reply briefly and "
    "warmly without forcing a knowledge-base answer.\n"
    "\n"
    "Rules:\n"
    "- Do not invent company policies, pricing, timelines, steps, guarantees, or "
    "service details.\n"
    "- Do not claim you checked information that is not present in the context.\n"
    "- Prefer short, clear answers.\n"
    "- When the answer is grounded in the KB, you may mention the relevant service "
    "name naturally if it helps.\n"
    "- Keep the conversation focused on COB Company services and related support topics.\n"
    "- Proactively share useful service-related details when they are available in context.\n"
    "- If the user asks about unrelated topics, gently redirect to COB Company topics.\n"
    "- Do not mention FAQ ids, vector search, retrieval, or internal system details "
    "unless the user explicitly asks.\n"
    "- Offer scheduling a meeting as an optional next step, not as the main response.\n"
    "- Prioritize answering the user's information needs first.\n"
    "- If the user appears satisfied or asks about next steps, then suggest booking a meeting.\n"
    "- After a longer multi-turn chat, start offering a meeting option as a helpful next step.\n"
    "- If needed information is not available in context, offer a meeting as an option for deeper support.\n"
    "- If the user sounds stressed or frustrated, respond calmly and offer a meeting or human follow-up option.\n"
    "\n"
    "Style:\n"
    "- Sound like a professional support assistant, not robotic.\n"
    "- Be interactive by inviting one useful follow-up question about services.\n"
    "- Keep meeting offers supportive and secondary unless the user asks to proceed.\n"
    "- Keep replies short by default: 2 short paragraphs max, around 60-120 words.\n"
    "- Format responses so they are organized and easy to read.\n"
    "- When listing services or steps, use short bullet points.\n"
    "- Use brief section labels if it improves readability.\n"
    "- When the context includes a company overview, include a short Summary section and a Services section listing the core service areas.\n"
    "- Be concise, friendly, and easy to understand."
)

COMPANY_OVERVIEW_REFERENCE = (
    "Company Overview Reference:\n"
    "COB Solution is a healthcare support company founded in 2020 by medical providers who "
    "understood the operational challenges outpatient practices face every day. The company "
    "focuses on helping healthcare organizations improve administrative efficiency, financial "
    "performance, patient support, and growth readiness through tailored business services.\n"
    "\n"
    "COB Solution presents itself as a trusted healthcare partner and highlights a fast "
    "onboarding process that can be completed in five steps: schedule a meeting, finalize the "
    "agreement, gather operational insights, choose or customize a package, and complete "
    "onboarding within 24 hours.\n"
    "\n"
    "COB Solution emphasizes practical experience in the medical field, customized support, "
    "and a broad service offering designed for outpatient healthcare providers. The company "
    "states that it has partnered with more than 100 medical locations and aims to help clients "
    "reduce operational friction while expanding sustainably.\n"
    "\n"
    "Core service areas:\n"
    "- Credentialing and provider maintenance\n"
    "- Authorizations\n"
    "- Benefits verification\n"
    "- Medical billing and denial management\n"
    "- Medical auditing\n"
    "- Financial management\n"
    "- Digital marketing and website services\n"
    "- Customer care\n"
    "- Elevate communication services\n"
    "\n"
    "Mission and growth vision:\n"
    "COB Solution aims to become a go-to partner for healthcare businesses and providers across "
    "the United States. Its stated purpose is to help healthcare organizations grow, simplify "
    "operations, and focus more time on patient care. The company describes its approach as "
    "transparent, personalized, and results-oriented, with an emphasis on long-term partnerships "
    "and measurable business improvement.\n"
    "\n"
    "Typical client profile:\n"
    "- Outpatient healthcare providers\n"
    "- Medical practices seeking operational support\n"
    "- Clinic owners exploring outsourced administrative services\n"
    "- Providers evaluating billing, credentialing, or patient support partners\n"
    "- Practices preparing for expansion or workflow modernization\n"
)


def build_kb_user_prompt(
    user_query: str,
    retrieved_context: list[str],
    conversation_history: list[str],
) -> str:
    history_block = build_history_block(conversation_history)
    context_block = "\n\n".join(retrieved_context)
    return (
        "Recent conversation:\n"
        f"{history_block}\n\n"
        "Customer question:\n"
        f"{user_query}\n\n"
        "Retrieved knowledge-base context:\n"
        f"{context_block or '[none]'}\n\n"
        f"{COMPANY_OVERVIEW_REFERENCE}\n\n"
        "Write the final answer for the customer in a natural, conversational way. "
        "Use only the information available in the retrieved context. "
        "Stay focused on COB Company topics. "
        "Format the reply so it is organized and easy to read; "
        "use short bullets for lists. "
        "When the retrieved context includes a company overview, add a short "
        "Summary section and a Services section listing the core service areas. "
        "Keep the reply concise (around 60-120 words). "
        "If the context is not enough, be honest and ask one short clarifying question."
    )


def build_history_block(conversation_history: list[str]) -> str:
    if not conversation_history:
        return "[no prior conversation]"

    normalized_messages = [
        message.strip() for message in conversation_history if message.strip()
    ]
    if not normalized_messages:
        return "[no prior conversation]"

    summary_line = ""
    if normalized_messages[0].startswith("summary:"):
        summary_line = normalized_messages[0]
        normalized_messages = normalized_messages[1:]

    recent_messages = normalized_messages[-6:]
    lines = [*recent_messages]
    if summary_line:
        lines = [summary_line, *recent_messages]
    return "\n".join(lines) if lines else "[no prior conversation]"
