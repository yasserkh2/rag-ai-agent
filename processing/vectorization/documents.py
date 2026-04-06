from __future__ import annotations

from processing.chunking.models import TextChunk
from processing.vectorization.contracts import (
    EmbeddingGenerator,
    VectorizationStrategy,
)
from processing.vectorization.models import VectorizationResult
from vector_db.models import VectorRecord


class DocumentVectorizationStrategy(VectorizationStrategy):
    def __init__(self, embedding_generator: EmbeddingGenerator) -> None:
        self._embedding_generator = embedding_generator

    def vectorize(self, chunks: list[TextChunk]) -> VectorizationResult:
        embeddings = self._embedding_generator.embed_documents(
            [chunk.text for chunk in chunks]
        )
        vector_records = [
            self._build_vector_record(chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        return VectorizationResult(
            records_processed=len(vector_records),
            vector_records=vector_records,
        )

    def _build_vector_record(
        self,
        chunk: TextChunk,
        embedding: list[float],
    ) -> VectorRecord:
        return VectorRecord(
            record_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            embedding=embedding,
        )
