from __future__ import annotations

import hashlib

from processing.vectorization.contracts import EmbeddingGenerator


class DeterministicEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, dimension: int = 16) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single_text(text) for text in texts]

    def _embed_single_text(self, text: str) -> list[float]:
        seed = text.encode("utf-8")
        values: list[float] = []
        counter = 0

        while len(values) < self._dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)
                if len(values) == self._dimension:
                    break
            counter += 1

        return values
