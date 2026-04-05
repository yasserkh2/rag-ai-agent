from __future__ import annotations

import json
import os
from urllib import error, request

from processing.vectorization.contracts import EmbeddingGenerator


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/embeddings",
        timeout_seconds: int = 60,
    ) -> None:
        if not model.strip():
            raise ValueError("Embedding model must not be empty.")
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY must not be empty.")

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenAIEmbeddingGenerator":
        return cls(
            model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            ),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps(
            {
                "model": self._model,
                "input": texts,
            }
        ).encode("utf-8")
        http_request = request.Request(
            self._base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self._timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI embeddings request failed with status {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"OpenAI embeddings request failed: {exc.reason}"
            ) from exc

        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings response did not contain data.")

        embeddings: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(
                    "OpenAI embeddings response contained an invalid embedding."
                )
            embeddings.append([float(value) for value in embedding])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "OpenAI embeddings response count did not match request count."
            )

        return embeddings
