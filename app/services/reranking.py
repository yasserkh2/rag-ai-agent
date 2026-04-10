from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from app.observability import get_logger, truncate_text
from vector_db.models import VectorSearchMatch

logger = get_logger("services.reranking")


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value.strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class CohereReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        url: str,
        timeout_seconds: float,
        max_documents: int,
        warmup_request: bool,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._max_documents = max_documents
        self._warmup_request = warmup_request

    def rerank(
        self,
        *,
        query: str,
        matches: list[VectorSearchMatch],
        top_k: int,
    ) -> list[VectorSearchMatch] | None:
        normalized_query = query.strip()
        if not normalized_query or not matches or top_k <= 0:
            return None

        documents: list[dict[str, Any]] = []
        match_lookup: list[VectorSearchMatch] = []
        for match in matches[: self._max_documents]:
            text = str(match.payload.get("text", "")).strip()
            if not text:
                text = self._fallback_text(match).strip()
            if not text:
                continue
            documents.append({"text": text})
            match_lookup.append(match)

        if not documents:
            return None

        payload = {
            "model": self._model,
            "query": normalized_query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }

        response = self._post_json(payload)
        results = response.get("results") if isinstance(response, dict) else None
        if not results:
            logger.warning(
                "cohere rerank returned no results for query='%s'",
                truncate_text(normalized_query, 100),
            )
            return None

        reranked: list[VectorSearchMatch] = []
        for item in results:
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(match_lookup):
                continue
            try:
                rerank_score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                rerank_score = 0.0

            match = match_lookup[index]
            payload = dict(match.payload)
            payload["vector_score"] = match.score
            payload["rerank_score"] = rerank_score
            reranked.append(replace(match, score=rerank_score, payload=payload))

        return reranked or None

    def warmup(self) -> None:
        if not self._api_key:
            raise RuntimeError("Cohere API key is missing.")
        if not self._model:
            raise RuntimeError("Cohere rerank model is missing.")
        if not self._url:
            raise RuntimeError("Cohere rerank URL is missing.")
        if not self._warmup_request:
            return

        payload = {
            "model": self._model,
            "query": "warmup",
            "documents": [{"text": "ping"}],
            "top_n": 1,
        }
        self._post_json(payload)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else str(exc)
            logger.warning("cohere rerank failed (%s): %s", exc.code, detail)
            raise RuntimeError("Cohere rerank request failed.") from exc
        except URLError as exc:
            logger.warning("cohere rerank unavailable: %s", exc.reason)
            raise RuntimeError("Cohere rerank unavailable.") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("cohere rerank returned invalid JSON")
            raise RuntimeError("Cohere rerank returned invalid JSON.") from exc

    def _fallback_text(self, match: VectorSearchMatch) -> str:
        payload = match.payload
        for key in ("title", "section_title", "service_name"):
            value = str(payload.get(key, "")).strip()
            if value:
                return value
        return ""


def build_reranker_from_env() -> CohereReranker | None:
    enabled = _parse_bool(os.getenv("RERANKER_ENABLED"), default=False)
    if not enabled:
        return None

    provider = os.getenv("RERANKER_PROVIDER", "cohere").strip().lower()
    if provider and provider not in {"cohere", "cohere_rerank"}:
        logger.warning("reranker provider '%s' is not supported", provider)
        return None

    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        logger.warning("reranker enabled but COHERE_API_KEY is missing")
        return None

    model = (
        os.getenv("RERANKER_MODEL")
        or os.getenv("COHERE_RERANK_MODEL")
        or "rerank-english-v3.0"
    ).strip()
    url = os.getenv("COHERE_RERANK_URL", "https://api.cohere.com/v1/rerank").strip()
    timeout_seconds = _parse_float(
        os.getenv("RERANKER_TIMEOUT_SECONDS"), default=8.0
    )
    max_documents = _parse_int(os.getenv("RERANKER_MAX_DOCUMENTS"), default=50)
    warmup_request = _parse_bool(
        os.getenv("RERANKER_WARMUP_REQUEST"), default=True
    )
    return CohereReranker(
        api_key=api_key,
        model=model,
        url=url,
        timeout_seconds=timeout_seconds,
        max_documents=max_documents,
        warmup_request=warmup_request,
    )


def rerank_candidate_limit(retrieval_limit: int) -> int:
    default_limit = max(6, retrieval_limit * 4)
    return _parse_int(os.getenv("RERANKER_CANDIDATES"), default=default_limit)


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        matches: list[VectorSearchMatch],
        top_k: int,
    ) -> list[VectorSearchMatch] | None: ...

    def warmup(self) -> None: ...
