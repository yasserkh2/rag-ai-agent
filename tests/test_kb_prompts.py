from __future__ import annotations

import unittest

from app.llm.prompts import DEFAULT_KB_SYSTEM_PROMPT, build_kb_user_prompt


class KnowledgeBasePromptTests(unittest.TestCase):
    def test_system_prompt_includes_unknown_answer_guardrails(self) -> None:
        self.assertIn(
            "clearly say you do not know based on the available information",
            DEFAULT_KB_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not use phrasing like \"Would you like to ...\" when information is missing or uncertain.",
            DEFAULT_KB_SYSTEM_PROMPT,
        )

    def test_user_prompt_includes_unknown_answer_guardrails(self) -> None:
        prompt = build_kb_user_prompt(
            user_query="I still do not understand",
            retrieved_context=["Document: doc_1\nText: sample"],
            conversation_history=["user: hi", "assistant: hello"],
        )
        self.assertIn(
            "If the context is not enough, clearly say you do not know from the current information",
            prompt,
        )
        self.assertIn(
            "Do not guess and do not use phrasing like 'Would you like to ...' when information is missing.",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
