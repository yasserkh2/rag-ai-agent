from __future__ import annotations

import os

from processing.vectorization.contracts import EmbeddingGenerator
from processing.vectorization.providers.gemini import GeminiEmbeddingGenerator
from processing.vectorization.providers.local import DeterministicEmbeddingGenerator
from processing.vectorization.providers.openai import OpenAIEmbeddingGenerator


class EmbeddingProviderFactory:
    def build(self, embedding_dimension: int) -> EmbeddingGenerator:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()

        if provider == "openai":
            return OpenAIEmbeddingGenerator.from_env()

        if provider == "gemini":
            return GeminiEmbeddingGenerator.from_env(
                default_output_dimensionality=embedding_dimension
            )

        if provider == "local":
            return DeterministicEmbeddingGenerator(embedding_dimension)

        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER "
            f"'{provider}'. Use 'openai', 'gemini', or 'local'."
        )


def build_embedding_generator(embedding_dimension: int) -> EmbeddingGenerator:
    return EmbeddingProviderFactory().build(embedding_dimension)
