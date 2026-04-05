from __future__ import annotations

from abc import ABC, abstractmethod

from processing.chunking.models import TextChunk
from processing.vectorization.models import VectorizationResult


class EmbeddingGenerator(ABC):
    def embed_text(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VectorizationStrategy(ABC):
    @abstractmethod
    def vectorize(self, chunks: list[TextChunk]) -> VectorizationResult:
        raise NotImplementedError
