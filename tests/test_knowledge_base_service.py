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


class RetrievalKnowledgeBaseServiceTests(unittest.TestCase):
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
        self.assertEqual(len(answer_generator.calls[0]["retrieved_context"]), 1)
        self.assertEqual(answer_generator.calls[0]["conversation_history"], [])

    def test_falls_back_to_extractive_answer_when_generation_fails(self) -> None:
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
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertEqual(
            result.final_response,
            "Credentialing includes primary source verification and application "
            "review.\n\nService: Credentialing | Source: FAQ faq_001",
        )
        self.assertEqual(result.turn_outcome, "resolved")

    def test_returns_no_match_message_when_search_is_empty(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=StubVectorSearcher([]),
            answer_generator=StubAnswerGenerator("unused"),
        )

        result = service.answer({"user_query": "Do you offer weekend support?"})

        self.assertIn("could not find a grounded answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "no_grounded_answer")
        self.assertEqual(list(result.retrieved_context), [])

    def test_returns_unavailable_message_when_retrieval_fails(self) -> None:
        service = RetrievalKnowledgeBaseService(
            embedding_generator=StubEmbeddingGenerator([1.0]),
            searcher=BrokenVectorSearcher(),
            answer_generator=StubAnswerGenerator("unused"),
        )

        result = service.answer({"user_query": "What is the status?"})

        self.assertIn("could not access the knowledge base", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(result.turn_failure_reason, "knowledge_base_unavailable")
        self.assertEqual(list(result.retrieved_context), [])

    def test_greeting_uses_conversational_generation_without_retrieval(self) -> None:
        embedding_generator = StubEmbeddingGenerator([1.0])
        searcher = StubVectorSearcher([])
        answer_generator = StubAnswerGenerator(
            "Hello! How can I help you with COB Company's services or policies today?"
        )
        service = RetrievalKnowledgeBaseService(
            embedding_generator=embedding_generator,
            searcher=searcher,
            answer_generator=answer_generator,
        )

        result = service.answer({"user_query": "Hello", "history": ["user: hi there"]})

        self.assertEqual(
            result.final_response,
            "Hello! How can I help you with COB Company's services or policies today?",
        )
        self.assertEqual(result.turn_outcome, "resolved")
        self.assertEqual(list(result.retrieved_context), [])
        self.assertEqual(embedding_generator.queries, [])
        self.assertEqual(searcher.calls, [])
        self.assertEqual(answer_generator.calls[0]["retrieved_context"], [])
        self.assertEqual(answer_generator.calls[0]["conversation_history"], ["user: hi there"])

    def test_filters_unrelated_retrieval_matches_before_answering(self) -> None:
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
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertIn("could not find a grounded answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(list(result.retrieved_context), [])

    def test_returns_document_context_and_fallback_for_document_match(self) -> None:
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
        )

        result = service.answer({"user_query": "What should we prepare before discussing this service?"})

        self.assertIn("Title: Authorizations and Benefits Verification", result.final_response)
        self.assertIn(
            "Source: Document doc_0002",
            result.final_response,
        )
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

    def test_filters_unrelated_document_matches_before_answering(self) -> None:
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
        )

        result = service.answer({"user_query": "What does credentialing include?"})

        self.assertIn("could not find a grounded answer", result.final_response)
        self.assertEqual(result.turn_outcome, "unresolved")
        self.assertEqual(list(result.retrieved_context), [])


if __name__ == "__main__":
    unittest.main()
