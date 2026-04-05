"""Vectorization package for turning text chunks into vector records."""

from processing.vectorization.contracts import EmbeddingGenerator, VectorizationStrategy
from processing.vectorization.factory import build_embedding_generator
from processing.vectorization.faqs import FaqVectorizationStrategy
from processing.vectorization.models import VectorizationResult
from processing.vectorization.providers import (
    DeterministicEmbeddingGenerator,
    EmbeddingProviderFactory,
    GeminiEmbeddingGenerator,
    OpenAIEmbeddingGenerator,
)

__all__ = [
    "build_embedding_generator",
    "DeterministicEmbeddingGenerator",
    "EmbeddingGenerator",
    "EmbeddingProviderFactory",
    "FaqVectorizationStrategy",
    "GeminiEmbeddingGenerator",
    "OpenAIEmbeddingGenerator",
    "VectorizationResult",
    "VectorizationStrategy",
]
