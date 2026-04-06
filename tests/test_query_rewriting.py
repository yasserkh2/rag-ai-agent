from __future__ import annotations

import unittest

from app.services.query_rewriting import DefaultRetrievalQueryRewriter


class DefaultRetrievalQueryRewriterTests(unittest.TestCase):
    def test_rewrites_service_scope_fragment_from_recent_history(self) -> None:
        rewriter = DefaultRetrievalQueryRewriter()

        rewritten = rewriter.rewrite(
            query="What This Service Usually Includes",
            history=[
                "user: do Credentialing and Provider Maintenance supports provider enrollment",
                "assistant: Yes, the Credentialing and Provider Maintenance service supports provider enrollment.",
            ],
        )

        self.assertEqual(
            rewritten,
            "What does Credentialing and Provider Maintenance usually include?",
        )

    def test_rewrites_escalation_fragment_from_recent_history(self) -> None:
        rewriter = DefaultRetrievalQueryRewriter()

        rewritten = rewriter.rewrite(
            query="When should that be escalated?",
            history=[
                "user: Tell me about Digital Marketing and Website Services",
                "assistant: Digital marketing and website services cover website support and visibility work.",
            ],
        )

        self.assertEqual(
            rewritten,
            "When should a conversation about Digital Marketing and Website Services be escalated to a human?",
        )

    def test_leaves_query_unchanged_without_recent_service_context(self) -> None:
        rewriter = DefaultRetrievalQueryRewriter()

        rewritten = rewriter.rewrite(
            query="What This Service Usually Includes",
            history=["user: hello", "assistant: Hi there!"],
        )

        self.assertEqual(rewritten, "What This Service Usually Includes")


if __name__ == "__main__":
    unittest.main()
