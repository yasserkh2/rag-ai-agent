from __future__ import annotations

import unittest

from app.services.knowledge_base import RetrievalKnowledgeBaseService
from vector_db.models import VectorSearchMatch


class StubEmbeddingGenerator:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self._vector)


class StubVectorSearcher:
    def __init__(self, matches: list[VectorSearchMatch]) -> None:
        self._matches = matches
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        with_vectors: bool = False,
    ) -> list[VectorSearchMatch]:
        self.calls.append(
            {
                "query_vector": list(query_vector),
                "limit": limit,
                "with_vectors": with_vectors,
            }
        )
        return list(self._matches)


class StubAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[dict[str, object]] = []

    def generate_answer(
        self,
        user_query: str,
        retrieved_context: list[str],
        conversation_history: list[str],
    ) -> str:
        self.calls.append(
            {
                "user_query": user_query,
                "retrieved_context": list(retrieved_context),
                "conversation_history": list(conversation_history),
            }
        )
        return self._answer


class BrokenVectorSearcher:
    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        with_vectors: bool = False,
    ) -> list[VectorSearchMatch]:
        raise RuntimeError("Qdrant is unavailable.")


class BrokenAnswerGenerator:
    def generate_answer(
        self,
        user_query: str,
        retrieved_context: list[str],
        conversation_history: list[str],
    ) -> str:
        raise RuntimeError("Generation failed.")


class StubQueryRewriter:
    def __init__(self, rewritten_query: str) -> None:
        self._rewritten_query = rewritten_query
        self.calls: list[dict[str, object]] = []

    def rewrite(self, query: str, history: list[str]) -> str:
        self.calls.append(
            {
                "query": query,
                "history": list(history),
            }
        )
        return self._rewritten_query


class IdentityQueryRewriter:
    def rewrite(self, query: str, history: list[str]) -> str:
        return query


class BrokenQueryRewriter:
    def rewrite(self, query: str, history: list[str]) -> str:
        raise RuntimeError("query generation failed")


class RetrievalKnowledgeBaseServiceTests(unittest.TestCase):
    def test_contact_query_prioritizes_chunk_with_contact_details(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            document_searcher=StubVectorSearcher(
                [
                    VectorSearchMatch(
                        point_id="point-doc-scenarios",
                        record_id="doc_0011_chunk_0002",
                        score=0.74,
                        payload={
                            "doc_id": "doc_0011",
                            "service_name": "COB Solution Contact Information",
                            "title": "COB Solution Contact Information",
                            "section_title": "Common Practice Scenarios",
                            "source_type": "document",
                            "text": (
                                "Title: COB Solution Contact Information\n"
                                "Service: COB Solution Contact Information\n"
                                "Section: Common Practice Scenarios\n\n"
                                "Common Practice Scenarios\n"
                                "- A user asks where COB Solution is located."
                            ),
                        },
                    ),
                    VectorSearchMatch(
                        point_id="point-doc-contact",
                        record_id="doc_0011_chunk_0001",
                        score=0.72,
                        payload={
                            "doc_id": "doc_0011",
                            "service_name": "COB Solution Contact Information",
                            "title": "COB Solution Contact Information",
                            "section_title": "Service Overview | What This Service Usually Includes",
                            "source_type": "document",
                            "text": (
                                "Title: COB Solution Contact Information\n"
                                "Service: COB Solution Contact Information\n"
                                "Section: Service Overview | What This Service Usually Includes\n\n"
                                "What This Service Usually Includes\n"
                                "- Phone: +1 (929) 229-7207\n"
                                "- Email: info@cobsolution.com\n"
                                "- Office address: Midtown, 575 8th Ave, New York, NY 10018"
                            ),
                        },
                    ),
                ]
            ),
            answer_generator=StubAnswerGenerator("unused"),
            retrieval_limit=1,
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "how can i contact the company?"})

        self.assertEqual(result.turn_outcome, "resolved")
        self.assertEqual(len(result.retrieved_context), 1)
        self.assertTrue(result.retrieved_context[0].startswith("Document: doc_0011"))
        self.assertIn("Phone: +1 (929) 229-7207", result.retrieved_context[0])

    def test_returns_grounded_generated_answer_and_context(self) -> None:
        embedding_generator = StubEmbeddingGenerator([0.1, 0.2, 0.3])
        searcher = StubVectorSearcher(
            [
                VectorSearchMatch(
                    point_id="point-1",
                    record_id="faq_001_chunk_0001",
                    score=0.93,
                    payload={
                        "faq_id": "faq_001",
                        "category": "credentialing",
                        "service_name": "Credentialing",
                        "text": (
                            "Question: What does credentialing include?\n"
                            "Answer: Credentialing includes primary source "
                            "verification and application review.\n"
                            "Service: Credentialing"
                        ),
                    },
                ),
                VectorSearchMatch(
                    point_id="point-2",
                    record_id="faq_010_chunk_0001",
                    score=0.71,
                    payload={
                        "faq_id": "faq_010",
                        "category": "enrollment",
                        "service_name": "Enrollment",
                        "text": (
                            "Question: How long does enrollment take?\n"
                            "Answer: Enrollment timelines vary by payer.\n"
                            "Service: Enrollment"
                        ),
                    },
                ),
            ]
        )
        answer_generator = StubAnswerGenerator(
            "Credentialing includes primary source verification and application "
            "review. This answer is based on the retrieved FAQ context."
        )
        service = RetrievalKnowledgeBaseService(
            embedding_generator=embedding_generator,
            searcher=searcher,
            answer_generator=answer_generator,
            retrieval_limit=2,
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertEqual(
            result.final_response,
            "Credentialing includes primary source verification and application "
            "review. This answer is based on the retrieved FAQ context.",
        )
        self.assertEqual(result.turn_outcome, "resolved")
        self.assertIsNone(result.turn_failure_reason)
        self.assertEqual(
            result.retrieved_context[0],
            "FAQ: faq_001\n"
            "Score: 0.9300\n"
            "Category: credentialing\n"
            "Service: Credentialing\n"
            "Question: What does credentialing include?\n"
            "Answer: Credentialing includes primary source verification and "
            "application review.",
        )
        self.assertEqual(embedding_generator.queries, ["What does credentialing include?"])
        self.assertEqual(
            searcher.calls,
            [
                {
                    "query_vector": [0.1, 0.2, 0.3],
                    "limit": 2,
                    "with_vectors": False,
                }
            ],
        )
        self.assertEqual(answer_generator.calls[0]["user_query"], "What does credentialing include?")
        self.assertEqual(len(answer_generator.calls[0]["retrieved_context"]), 2)
        self.assertEqual(answer_generator.calls[0]["conversation_history"], [])

    def test_retrieves_using_raw_user_query(self) -> None:
        embedding_generator = StubEmbeddingGenerator([0.1, 0.2, 0.3])
        searcher = StubVectorSearcher(
            [
                VectorSearchMatch(
                    point_id="point-1",
                    record_id="faq_001_chunk_0001",
                    score=0.93,
                    payload={
                        "faq_id": "faq_001",
                        "category": "credentialing",
                        "service_name": "Credentialing and Provider Maintenance",
                        "text": (
                            "Question: What does credentialing include?\n"
                            "Answer: Credentialing includes primary source "
                            "verification and application review.\n"
                            "Service: Credentialing and Provider Maintenance"
                        ),
                    },
                )
            ]
        )
        query_rewriter = StubQueryRewriter(
            "What does Credentialing and Provider Maintenance usually include?"
        )
        service = RetrievalKnowledgeBaseService(
            embedding_generator=embedding_generator,
            searcher=searcher,
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=query_rewriter,
        )

        service.answer(
            {
                "user_query": "What This Service Usually Includes",
                "history": [
                    "user: do Credentialing and Provider Maintenance supports provider enrollment",
                    "assistant: Yes, the Credentialing and Provider Maintenance service supports provider enrollment.",
                ],
            }
        )

        self.assertEqual(
            embedding_generator.queries,
            ["What does Credentialing and Provider Maintenance usually include?"],
        )
        self.assertEqual(
            query_rewriter.calls,
            [
                {
                    "query": "What This Service Usually Includes",
                    "history": [
                        "user: do Credentialing and Provider Maintenance supports provider enrollment",
                        "assistant: Yes, the Credentialing and Provider Maintenance service supports provider enrollment.",
                    ],
                }
            ],
        )

    def test_returns_generation_error_when_answer_generation_fails(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher(
                [
                    VectorSearchMatch(
                        point_id="point-1",
                        record_id="faq_001_chunk_0001",
                        score=0.93,
                        payload={
                            "faq_id": "faq_001",
                            "category": "credentialing",
                            "service_name": "Credentialing",
                            "text": (
                                "Question: What does credentialing include?\n"
                                "Answer: Credentialing includes primary source "
                                "verification and application review.\n"
                                "Service: Credentialing"
                            ),
                        },
                    )
                ]
            ),
            answer_generator=BrokenAnswerGenerator(),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertIn("could not generate a reliable answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "answer_generation_failed")
        self.assertEqual(len(result.retrieved_context), 1)

    def test_returns_no_match_message_when_search_is_empty(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "Do you offer weekend support?"})

        self.assertIn("could not find a grounded answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "no_grounded_answer")
        self.assertEqual(list(result.retrieved_context), [])

    def test_returns_generation_error_for_document_when_answer_generation_fails(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher(
                [
                    VectorSearchMatch(
                        point_id="point-doc-1",
                        record_id="doc_0001_chunk_0001",
                        score=0.88,
                        payload={
                            "doc_id": "doc_0001",
                            "service_name": "COB Solution Company Overview",
                            "title": "COB Solution Company Overview",
                            "section_title": "How The Chatbot Should Respond",
                            "text": (
                                "Title: COB Solution Company Overview\n"
                                "Service: COB Solution Company Overview\n"
                                "Section: How The Chatbot Should Respond\n\n"
                                "How The Chatbot Should Respond\n"
                                "When a user asks about COB Solution generally, "
                                "the chatbot should provide a concise company overview "
                                "and mention the major service categories.\n\n"
                                "Example Customer Questions\n"
                                "- What is COB Solution?\n"
                                "- What services does COB Solution offer?\n"
                            ),
                        },
                    )
                ]
            ),
            answer_generator=BrokenAnswerGenerator(),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What services does COB Solution offer?"})

        self.assertIn("could not generate a reliable answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "answer_generation_failed")
        self.assertEqual(len(result.retrieved_context), 1)

    def test_returns_unavailable_message_when_retrieval_fails(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=BrokenVectorSearcher(),
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What is the status?"})

        self.assertIn("could not access the knowledge base", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "knowledge_base_unavailable")
        self.assertEqual(list(result.retrieved_context), [])

    def test_returns_error_when_query_generation_fails(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=BrokenQueryRewriter(),
        )

        result = service.answer({"user_query": "What is COB Solution?"})

        self.assertIn("could not prepare a reliable search query", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(
            result.turn_failure_reason,
            "retrieval_query_generation_failed",
        )

    def test_greeting_still_uses_retrieval_when_kb_service_is_called_directly(self) -> None:
        embedding_generator = StubEmbeddingGenerator([1.0])
        searcher = StubVectorSearcher([])
        service = RetrievalKnowledgeBaseService(
            embedding_generator=embedding_generator,
            searcher=searcher,
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "Hello", "history": ["user: hi there"]})

        self.assertIn("could not find a grounded answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "no_grounded_answer")
        self.assertEqual(list(result.retrieved_context), [])
        self.assertEqual(embedding_generator.queries, ["Hello"])
        self.assertEqual(len(searcher.calls), 1)

    def test_uses_retrieval_matches_without_extra_relevance_filtering(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher(
                [
                    VectorSearchMatch(
                        point_id="point-1",
                        record_id="faq_08750_chunk_0001",
                        score=0.95,
                        payload={
                            "faq_id": "faq_08750",
                            "category": "marketing",
                            "service_name": "Digital Marketing and Website Services",
                            "text": (
                                "Question: Do you support single-site and multi-location practices?\n"
                                "Answer: Yes. The mock dataset assumes support for single-site "
                                "and multi-location practices for Digital Marketing and Website "
                                "Services.\n"
                                "Service: Digital Marketing and Website Services"
                            ),
                        },
                    )
                ]
            ),
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertEqual(result.final_response, "unused")
        self.assertEqual(result.turn_outcome, "resolved")
        self.assertEqual(len(result.retrieved_context), 1)

    def test_returns_document_context_and_generation_error_for_document_match(self) -> None:
        document_searcher = StubVectorSearcher(
            [
                VectorSearchMatch(
                    point_id="point-doc-1",
                    record_id="doc_0002_chunk_0003",
                    score=0.94,
                    payload={
                        "doc_id": "doc_0002",
                        "service_name": "Authorizations and Benefits Verification",
                        "title": "Authorizations and Benefits Verification",
                        "section_title": "Information To Collect During Intake",
                        "source_type": "document",
                        "text": (
                            "Title: Authorizations and Benefits Verification\n"
                            "Service: Authorizations and Benefits Verification\n"
                            "Section: Information To Collect During Intake\n\n"
                            "Information To Collect During Intake\n"
                            "- Practice name\n"
                            "- Payer details\n"
                            "- Procedure types"
                        ),
                    },
                )
            ]
        )
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            document_searcher=document_searcher,
            answer_generator=BrokenAnswerGenerator(),
            retrieval_limit=2,
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What should we prepare before discussing this service?"})

        self.assertIn("could not generate a reliable answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "answer_generation_failed")
        self.assertEqual(
            result.retrieved_context[0],
            "Document: doc_0002\n"
            "Score: 0.9400\n"
            "Service: Authorizations and Benefits Verification\n"
            "Title: Authorizations and Benefits Verification\n"
            "Section: Information To Collect During Intake\n"
            "Text: Title: Authorizations and Benefits Verification\n"
            "Service: Authorizations and Benefits Verification\n"
            "Section: Information To Collect During Intake\n\n"
            "Information To Collect During Intake\n"
            "- Practice name\n"
            "- Payer details\n"
            "- Procedure types",
        )
        self.assertEqual(len(document_searcher.calls), 1)

    def test_merges_faq_and_document_results_for_generation(self) -> None:
        embedding_generator = StubEmbeddingGenerator([0.5, 0.6])
        faq_searcher = StubVectorSearcher(
            [
                VectorSearchMatch(
                    point_id="faq-point-1",
                    record_id="faq_001_chunk_0001",
                    score=0.91,
                    payload={
                        "faq_id": "faq_001",
                        "category": "service_scope",
                        "service_name": "Credentialing and Provider Maintenance",
                        "source_type": "faq",
                        "text": (
                            "Question: What does credentialing include?\n"
                            "Answer: Credentialing includes enrollment and provider maintenance.\n"
                            "Service: Credentialing and Provider Maintenance"
                        ),
                    },
                )
            ]
        )
        document_searcher = StubVectorSearcher(
            [
                VectorSearchMatch(
                    point_id="doc-point-1",
                    record_id="doc_0001_chunk_0003",
                    score=0.92,
                    payload={
                        "doc_id": "doc_0001",
                        "service_name": "Credentialing and Provider Maintenance",
                        "title": "Credentialing and Provider Maintenance",
                        "section_title": "Information To Collect During Intake",
                        "source_type": "document",
                        "text": (
                            "Title: Credentialing and Provider Maintenance\n"
                            "Service: Credentialing and Provider Maintenance\n"
                            "Section: Information To Collect During Intake\n\n"
                            "Information To Collect During Intake\n"
                            "- Practice name\n"
                            "- Provider names"
                        ),
                    },
                )
            ]
        )
        answer_generator = StubAnswerGenerator(
            "Credentialing includes enrollment support, and the intake details include "
            "practice and provider information."
        )
        service = RetrievalKnowledgeBaseService(
            embedding_generator=embedding_generator,
            searcher=faq_searcher,
            document_searcher=document_searcher,
            answer_generator=answer_generator,
            retrieval_limit=2,
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What does credentialing include and what should we prepare?"})

        self.assertEqual(result.turn_outcome, "resolved")
        self.assertEqual(len(result.retrieved_context), 2)
        self.assertTrue(result.retrieved_context[0].startswith("Document: doc_0001"))
        self.assertTrue(result.retrieved_context[1].startswith("FAQ: faq_001"))
        self.assertEqual(embedding_generator.queries, ["What does credentialing include and what should we prepare?"])
        self.assertEqual(len(faq_searcher.calls), 1)
        self.assertEqual(len(document_searcher.calls), 1)
        self.assertEqual(len(answer_generator.calls[0]["retrieved_context"]), 2)

    def test_uses_document_matches_without_extra_relevance_filtering(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            document_searcher=StubVectorSearcher(
                [
                    VectorSearchMatch(
                        point_id="point-doc-9",
                        record_id="doc_0006_chunk_0001",
                        score=0.95,
                        payload={
                            "doc_id": "doc_0006",
                            "service_name": "Digital Marketing and Website Services",
                            "title": "Digital Marketing and Website Services",
                            "section_title": "Service Overview | What This Service Usually Includes",
                            "source_type": "document",
                            "text": (
                                "Title: Digital Marketing and Website Services\n"
                                "Service: Digital Marketing and Website Services\n"
                                "Section: Service Overview | What This Service Usually Includes\n\n"
                                "Service Overview\n"
                                "This service helps practices improve online presence."
                            ),
                        },
                    )
                ]
            ),
            answer_generator=StubAnswerGenerator("unused"),
            query_rewriter=IdentityQueryRewriter(),
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertEqual(result.final_response, "unused")
        self.assertEqual(result.turn_outcome, "resolved")
        self.assertEqual(len(result.retrieved_context), 1)


if __name__ == "__main__":
    unittest.main()
