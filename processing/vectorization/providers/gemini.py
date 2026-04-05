from __future__ import annotations

import json
import os
from urllib import error, parse, request

from processing.vectorization.contracts import EmbeddingGenerator


class GeminiEmbeddingGenerator(EmbeddingGenerator):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: int = 60,
        output_dimensionality: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Gemini embedding model must not be empty.")
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY must not be empty.")
        if output_dimensionality is not None and output_dimensionality <= 0:
            raise ValueError("Gemini output dimensionality must be greater than zero.")

        self._model = self._normalize_model_name(model)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._output_dimensionality = output_dimensionality

    @classmethod
    def from_env(
        cls,
        default_output_dimensionality: int | None = None,
    ) -> "GeminiEmbeddingGenerator":
        raw_dimension = os.getenv("GEMINI_OUTPUT_DIMENSION", "").strip()
        output_dimensionality = (
            int(raw_dimension) if raw_dimension else default_output_dimensionality
        )
        return cls(
            model=os.getenv(
                "GEMINI_EMBEDDING_MODEL",
                os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            ),
            api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            output_dimensionality=output_dimensionality,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._batch_embed(texts=texts, task_type="RETRIEVAL_QUERY")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._batch_embed(texts=texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._batch_embed(texts=texts, task_type="SEMANTIC_SIMILARITY")

    def _batch_embed(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        if not texts:
            return []

        payload = {"requests": [self._build_request(text, task_type) for text in texts]}
        endpoint = (
            f"{self._base_url}/{self._model}:batchEmbedContents"
            f"?{parse.urlencode({'key': self._api_key})}"
        )
        response_payload = self._post_json(url=endpoint, payload=payload)

        embeddings = response_payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Gemini embeddings response did not contain embeddings.")

        values_list: list[list[float]] = []
        for item in embeddings:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "Gemini embeddings response contained an invalid embedding object."
                )

            values = item.get("values")
            if not isinstance(values, list):
                raise RuntimeError(
                    "Gemini embeddings response contained an invalid values list."
                )
            values_list.append([float(value) for value in values])

        if len(values_list) != len(texts):
            raise RuntimeError(
                "Gemini embeddings response count did not match request count."
            )

        return values_list

    def _build_request(self, text: str, task_type: str) -> dict[str, object]:
        embed_request: dict[str, object] = {
            "model": self._model,
            "content": {
                "parts": [{"text": text}],
            },
            "taskType": task_type,
        }

        if self._output_dimensionality is not None:
            embed_request["outputDimensionality"] = self._output_dimensionality

        return embed_request

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        http_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self._timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini embeddings request failed with status {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Gemini embeddings request failed: {exc.reason}"
            ) from exc

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        model_name = model.strip()
        if model_name.startswith("models/"):
            return model_name
        return f"models/{model_name}"
