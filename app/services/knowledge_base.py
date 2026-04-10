from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.graph.state import ChatState
from app.llm import AnswerGenerator, KbAnswerGeneratorFactory
from app.observability import get_logger, truncate_text
from app.services.contracts import RetrievalQueryRewriter
from app.services.models import KnowledgeBaseAnswer
from app.services.query_rewriting import LlmRetrievalQueryRewriter
from app.services.reranking import (
    Reranker,
    build_reranker_from_env,
    rerank_candidate_limit,
)
from processing.vectorization import build_embedding_generator
from processing.vectorization.contracts import EmbeddingGenerator
from vector_db.contracts import VectorSearcher
from vector_db.models import VectorSearchMatch

if TYPE_CHECKING:
    from vector_db.qdrant import QdrantSettings

logger = get_logger("services.knowledge_base")

_FAQ_TEXT_PATTERN = re.compile(
    r"^Question:\s*(?P<question>.*?)\n"
    r"Answer:\s*(?P<answer>.*?)\n"
    r"Service:\s*(?P<service>[^\n]*)",
    re.DOTALL,
)
_CONTACT_QUERY_TERMS = (
    "contact",
    "phone",
    "email",
    "e-mail",
    "address",
    "location",
    "where",
    "linkedin",
    "facebook",
    "instagram",
    "reach",
    "call",
)
_CONTACT_EVIDENCE_TERMS = (
    "phone:",
    "email:",
    "office address:",
    "map link:",
    "linkedin:",
    "facebook:",
    "instagram:",
)


@dataclass(frozen=True, slots=True)
class RetrievedContextItem:
    source_type: str
    record_id: str
    source_id: str
    score: float
    service: str
    title: str
    section_title: str
    category: str
    question: str
    answer: str
    raw_text: str
    vector_score: float

    def as_retrieved_context(self) -> str:
        if self.source_type == "document":
            lines = [f"Document: {self.source_id}", f"Score: {self.score:.4f}"]
            if self.service:
                lines.append(f"Service: {self.service}")
            if self.title:
                lines.append(f"Title: {self.title}")
            if self.section_title:
                lines.append(f"Section: {self.section_title}")
            if self.raw_text:
                lines.append(f"Text: {self.raw_text}")
            return "\n".join(lines)

        lines = [f"FAQ: {self.source_id}", f"Score: {self.score:.4f}"]
        if self.category:
            lines.append(f"Category: {self.category}")
        if self.service:
            lines.append(f"Service: {self.service}")
        if self.question:
            lines.append(f"Question: {self.question}")
        if self.answer:
            lines.append(f"Answer: {self.answer}")
        elif self.raw_text:
            lines.append(f"Text: {self.raw_text}")
        return "\n".join(lines)


class RetrievalKnowledgeBaseService:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        searcher: VectorSearcher | None = None,
        answer_generator: AnswerGenerator | None = None,
        retrieval_limit: int = 3,
        document_searcher: VectorSearcher | None = None,
        query_rewriter: RetrievalQueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero.")

        self._embedding_generator = embedding_generator
        self._searcher = searcher
        self._document_searcher = document_searcher
        self._answer_generator = answer_generator
        self._retrieval_limit = retrieval_limit
        self._search_documents = document_searcher is not None or searcher is None
        self._query_rewriter = query_rewriter or LlmRetrievalQueryRewriter()
        self._reranker = reranker or build_reranker_from_env()
        self._rerank_candidates = rerank_candidate_limit(self._retrieval_limit)
        self._settings: QdrantSettings | None = None
        self._document_settings: QdrantSettings | None = None
        self._warmed_up = False

    def warmup(self) -> None:
        if self._warmed_up:
            return

        warmup_start = perf_counter()
        try:
            self._get_embedding_generator()
            self._get_searcher()
            if self._search_documents:
                self._get_document_searcher()
            self._get_answer_generator()
            if self._reranker is not None:
                self._reranker.warmup()
            self._warmed_up = True
            logger.info(
                "kb service warmup completed (ms=%.1f)",
                (perf_counter() - warmup_start) * 1000,
            )
        except Exception as exc:
            logger.warning("kb service warmup skipped due to error: %s", exc)

    def answer(self, state: ChatState) -> KnowledgeBaseAnswer:
        total_start = perf_counter()
        query = state.get("user_query", "").strip()
        logger.info("kb service received query='%s'", truncate_text(query, 120))
        if not query:
            return KnowledgeBaseAnswer(
                final_response=(
                    "Please share your question and I will look for the closest "
                    "knowledge-base answer."
                ),
                retrieval_query="",
                turn_outcome="needs_input",
            )

        history = list(state.get("history", []))

        rewrite_start = perf_counter()
        try:
            retrieval_query = self._query_rewriter.rewrite(query=query, history=history)
        except Exception as exc:
            logger.warning("kb query rewriting failed: %s", exc)
            return KnowledgeBaseAnswer(
                final_response=(
                    "I could not prepare a reliable search query for your request. "
                    "Please try rephrasing your question."
                ),
                retrieval_query=query,
                turn_outcome="unresolved",
                turn_failure_reason="retrieval_query_generation_failed",
            )
        rewrite_ms = (perf_counter() - rewrite_start) * 1000
        logger.info("kb retrieval query='%s'", truncate_text(retrieval_query, 140))
        try:
            matches = self._retrieve(retrieval_query)
        except Exception as exc:
            logger.exception("kb retrieval failed: %s", exc)
            return KnowledgeBaseAnswer(
                final_response=(
                    "I could not access the knowledge base just now. "
                    "Please try again after the retrieval setup is ready."
                ),
                retrieval_query=retrieval_query,
                turn_outcome="unresolved",
                turn_failure_reason="knowledge_base_unavailable",
            )

        if not matches:
            logger.info("kb retrieval returned no matches")
            return KnowledgeBaseAnswer(
                final_response=(
                    "I could not find a grounded answer in the knowledge base yet. "
                    "Please rephrase your question or share a little more detail."
                ),
                retrieval_query=retrieval_query,
                turn_outcome="unresolved",
                turn_failure_reason="no_grounded_answer",
            )

        context_items = [self._build_context_item(match) for match in matches]

        retrieved_context = [
            context_item.as_retrieved_context() for context_item in context_items
        ]
        generation_start = perf_counter()
        try:
            final_response = self._generate_answer(
                user_query=query,
                retrieved_context=retrieved_context,
                conversation_history=history,
            )
        except Exception as exc:
            logger.warning("kb generation failed, returning explicit error reply: %s", exc)
            return KnowledgeBaseAnswer(
                final_response=(
                    "I found relevant information, but I could not generate a reliable "
                    "answer right now. Please try again."
                ),
                retrieval_query=retrieval_query,
                retrieved_context=retrieved_context,
                turn_outcome="unresolved",
                turn_failure_reason="answer_generation_failed",
            )
        generation_ms = (perf_counter() - generation_start) * 1000
        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            "kb final response ready with %s retrieved context items (rewrite_ms=%.1f generation_ms=%.1f total_ms=%.1f)",
            len(retrieved_context),
            rewrite_ms,
            generation_ms,
            total_ms,
        )
        return KnowledgeBaseAnswer(
            final_response=final_response,
            retrieval_query=retrieval_query,
            retrieved_context=retrieved_context,
            turn_outcome="resolved",
        )

    def _retrieve(self, query: str) -> list[VectorSearchMatch]:
        embedding_generator = self._get_embedding_generator()
        embed_start = perf_counter()
        query_vector = embedding_generator.embed_query(query)
        embed_ms = (perf_counter() - embed_start) * 1000
        logger.info("kb embedding generated for retrieval query")

        retrieval_start = perf_counter()
        searcher_setup_start = perf_counter()
        faq_searcher = self._get_searcher()
        document_searcher = self._get_document_searcher() if self._search_documents else None
        searcher_setup_ms = (perf_counter() - searcher_setup_start) * 1000
        faq_matches: list[VectorSearchMatch] = []
        faq_search_ms = 0.0
        document_matches: list[VectorSearchMatch] = []
        document_search_ms = 0.0

        def run_search(searcher: VectorSearcher) -> tuple[list[VectorSearchMatch], float]:
            search_start = perf_counter()
            matches = searcher.search(
                query_vector=query_vector,
                limit=self._retrieval_limit,
                with_vectors=False,
            )
            return matches, (perf_counter() - search_start) * 1000

        if self._search_documents:
            with ThreadPoolExecutor(max_workers=2) as executor:
                faq_future = executor.submit(run_search, faq_searcher)
                document_future = executor.submit(
                    run_search,
                    document_searcher,
                )
                faq_matches, faq_search_ms = faq_future.result()
                document_matches, document_search_ms = document_future.result()
        else:
            faq_matches, faq_search_ms = run_search(faq_searcher)

        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        raw_parallel_wait_ms = retrieval_ms - searcher_setup_ms - max(
            faq_search_ms,
            document_search_ms,
        )
        parallel_wait_overhead_ms = max(0.0, raw_parallel_wait_ms)
        combined_matches = faq_matches + document_matches
        combined_matches.sort(
            key=lambda item: self._scored_retrieval_priority(query, item),
            reverse=True,
        )
        top_matches = combined_matches[: self._retrieval_limit]
        if self._reranker and combined_matches:
            rerank_candidates = combined_matches[: self._rerank_candidates]
            rerank_start = perf_counter()
            try:
                reranked = self._reranker.rerank(
                    query=query,
                    matches=rerank_candidates,
                    top_k=self._retrieval_limit,
                )
            except Exception as exc:
                logger.warning("kb rerank failed: %s", exc)
                reranked = None
            rerank_ms = (perf_counter() - rerank_start) * 1000
            if reranked:
                top_matches = reranked
                logger.info(
                    "kb rerank applied candidates=%s results=%s rerank_ms=%.1f",
                    len(rerank_candidates),
                    len(reranked),
                    rerank_ms,
                )
            else:
                logger.info(
                    "kb rerank skipped candidates=%s rerank_ms=%.1f",
                    len(rerank_candidates),
                    rerank_ms,
                )
        logger.info(
            "kb retrieved faq_matches=%s document_matches=%s total=%s top=%s (embed_ms=%.1f searcher_setup_ms=%.1f faq_search_ms=%.1f doc_search_ms=%.1f retrieval_ms=%.1f parallel_wait_overhead_ms=%.1f)",
            len(faq_matches),
            len(document_matches),
            len(combined_matches),
            [
                {
                    "record_id": match.record_id,
                    "score": round(match.score, 4),
                    "source_type": str(match.payload.get("source_type", "")),
                }
                for match in top_matches
            ],
            embed_ms,
            searcher_setup_ms,
            faq_search_ms,
            document_search_ms,
            retrieval_ms,
            parallel_wait_overhead_ms,
        )
        return top_matches

    def _scored_retrieval_priority(
        self,
        query: str,
        match: VectorSearchMatch,
    ) -> tuple[float, float]:
        bonus = 0.0
        if self._is_contact_query(query):
            bonus = self._contact_evidence_bonus(match)
        return (match.score + bonus, match.score)

    def _is_contact_query(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        return any(term in normalized for term in _CONTACT_QUERY_TERMS)

    def _contact_evidence_bonus(self, match: VectorSearchMatch) -> float:
        payload = match.payload
        searchable = " ".join(
            str(payload.get(field, "")).strip().lower()
            for field in ("title", "service_name", "section_title", "text")
        )
        if not searchable:
            return 0.0

        evidence_hits = sum(
            1 for term in _CONTACT_EVIDENCE_TERMS if term in searchable
        )
        if evidence_hits == 0:
            return 0.0

        # Keep bonus small so vector similarity remains dominant while preferring
        # concrete contact-detail chunks when scores are close.
        return min(0.05, evidence_hits * 0.01)

    def _generate_answer(
        self,
        user_query: str,
        retrieved_context: list[str],
        conversation_history: list[str],
    ) -> str:
        generator = self._get_answer_generator()

        generated_answer = generator.generate_answer(
            user_query=user_query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_history,
        ).strip()

        if not generated_answer:
            raise RuntimeError("KB answer generator returned an empty response.")

        return generated_answer

    def _get_embedding_generator(self) -> EmbeddingGenerator:
        if self._embedding_generator is None:
            settings = self._get_settings()
            self._embedding_generator = build_embedding_generator(
                settings.embedding_dimension
            )
        return self._embedding_generator

    def _get_searcher(self) -> VectorSearcher:
        if self._searcher is None:
            from vector_db.qdrant import QdrantVectorSearcher

            self._searcher = QdrantVectorSearcher(settings=self._get_settings())
        return self._searcher

    def _get_document_searcher(self) -> VectorSearcher:
        if self._document_searcher is None:
            from vector_db.qdrant import QdrantVectorSearcher

            self._document_searcher = QdrantVectorSearcher(
                settings=self._get_document_settings()
            )
        return self._document_searcher

    def _get_answer_generator(self) -> AnswerGenerator | None:
        if self._answer_generator is None:
            self._answer_generator = KbAnswerGeneratorFactory().build()
        return self._answer_generator

    def _get_settings(self) -> QdrantSettings:
        if self._settings is None:
            from vector_db.qdrant import QdrantSettings

            self._settings = QdrantSettings.from_env()
        return self._settings

    def _get_document_settings(self) -> QdrantSettings:
        if self._document_settings is None:
            from vector_db.qdrant import QdrantSettings

            self._document_settings = QdrantSettings.from_env(
                collection_env_key="QDRANT_DOCUMENT_COLLECTION",
                collection_default="customer_care_documents_kb",
            )
        return self._document_settings

    def _build_context_item(self, match: VectorSearchMatch) -> RetrievedContextItem:
        payload = match.payload
        payload_text = str(payload.get("text", "")).strip()
        source_type = str(payload.get("source_type", "")).strip().lower()
        if source_type not in {"faq", "document"}:
            source_type = "document" if "doc_id" in payload else "faq"

        if source_type == "document":
            doc_id = str(payload.get("doc_id", "")).strip() or match.record_id
            return RetrievedContextItem(
                source_type="document",
                record_id=match.record_id,
                source_id=doc_id,
                score=match.score,
                service=str(payload.get("service_name", "")).strip(),
                title=str(payload.get("title", "")).strip(),
                section_title=str(payload.get("section_title", "")).strip(),
                category="",
                question="",
                answer="",
                raw_text=payload_text,
                vector_score=match.score,
            )

        parsed = _FAQ_TEXT_PATTERN.match(payload_text)
        if parsed is None:
            question = ""
            answer = payload_text
            service = str(payload.get("service_name", "")).strip()
        else:
            question = parsed.group("question").strip()
            answer = parsed.group("answer").strip()
            service = parsed.group("service").strip()

        faq_id = str(payload.get("faq_id", "")).strip() or match.record_id
        category = str(payload.get("category", "")).strip()
        return RetrievedContextItem(
            source_type="faq",
            record_id=match.record_id,
            source_id=faq_id,
            score=match.score,
            service=service,
            title="",
            section_title="",
            category=category,
            question=question,
            answer=answer,
            raw_text=payload_text,
            vector_score=match.score,
        )
