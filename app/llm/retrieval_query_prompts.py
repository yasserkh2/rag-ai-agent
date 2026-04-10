from __future__ import annotations


DEFAULT_RETRIEVAL_QUERY_SYSTEM_PROMPT = (
    "You rewrite customer chat turns into one optimized retrieval query for a vector "
    "database.\n"
    "Rewrite for vector search to retrieve the right chunks.\n"
    "Use the latest user message and recent chat history.\n"
    "Return only the rewritten query text.\n"
    "Do not add explanations, labels, bullets, or quotes.\n"
    "Keep service names and key entities exact when present.\n"
    "Resolve pronouns like he/she/it/they to the specific entity when possible.\n"
    "Do not add extra context; keep it simple and short.\n"
    "If the latest user message is already clear, return it unchanged."
)


def build_retrieval_query_prompt(
    user_query: str,
    conversation_history: list[str],
) -> str:
    history_block = "\n".join(conversation_history[-8:]) or "[no prior conversation]"
    return (
        "Rewrite the customer message into a standalone retrieval query.\n\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"Latest user message:\n{user_query}\n\n"
        "Return the rewritten retrieval query only."
    )
