from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from vector_db.contracts import VectorDatabaseSetup
from vector_db.models import VectorCollectionSetupResult

_CLIENT_CACHE: dict[tuple[str, ...], QdrantClient] = {}


def _close_cached_clients() -> None:
    unique_clients = {id(client): client for client in _CLIENT_CACHE.values()}
    for client in unique_clients.values():
        try:
            client.close()
        except Exception:
            pass


atexit.register(_close_cached_clients)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    collection_name: str
    embedding_dimension: int
    storage_path: Path
    distance: Distance
    prefer_grpc: bool = False
    url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        collection_env_key: str = "QDRANT_COLLECTION",
        collection_default: str = "customer_care_kb",
    ) -> "QdrantSettings":
        distance_name = os.getenv("QDRANT_DISTANCE", "cosine").strip().upper()
        try:
            distance = Distance[distance_name]
        except KeyError as exc:
            supported = ", ".join(item.name.lower() for item in Distance)
            raise ValueError(
                f"Unsupported QDRANT_DISTANCE '{distance_name.lower()}'. "
                f"Use one of: {supported}."
            ) from exc

        return cls(
            collection_name=os.getenv(collection_env_key, collection_default),
            embedding_dimension=int(os.getenv("QDRANT_EMBEDDING_DIMENSION", "1536")),
            storage_path=Path(
                os.getenv("QDRANT_PATH", "vector_db/qdrant/data/local")
            ),
            distance=distance,
            prefer_grpc=_parse_bool(os.getenv("QDRANT_PREFER_GRPC")),
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )


class QdrantVectorDatabaseSetup(VectorDatabaseSetup):
    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self._client: QdrantClient | None = None

    @property
    def settings(self) -> QdrantSettings:
        return self._settings

    def create_client(self) -> QdrantClient:
        if self._client is not None:
            return self._client

        if self._settings.url:
            cache_key = (
                "url",
                self._settings.url,
                self._settings.api_key or "",
                str(self._settings.prefer_grpc),
            )
            cached_client = _CLIENT_CACHE.get(cache_key)
            if cached_client is None:
                cached_client = QdrantClient(
                    url=self._settings.url,
                    api_key=self._settings.api_key,
                    prefer_grpc=self._settings.prefer_grpc,
                )
                _CLIENT_CACHE[cache_key] = cached_client
            self._client = cached_client
            return self._client

        self._settings.storage_path.mkdir(parents=True, exist_ok=True)
        cache_key = ("path", str(self._settings.storage_path.resolve()))
        cached_client = _CLIENT_CACHE.get(cache_key)
        if cached_client is None:
            cached_client = QdrantClient(path=str(self._settings.storage_path))
            _CLIENT_CACHE[cache_key] = cached_client
        self._client = cached_client
        return self._client

    def ensure_collection(self) -> VectorCollectionSetupResult:
        client = self.create_client()

        if not client.collection_exists(self._settings.collection_name):
            client.create_collection(
                collection_name=self._settings.collection_name,
                vectors_config=VectorParams(
                    size=self._settings.embedding_dimension,
                    distance=self._settings.distance,
                ),
            )
            created = True
        else:
            created = False

        return VectorCollectionSetupResult(
            collection_name=self._settings.collection_name,
            created=created,
            backend=self._settings.url or str(self._settings.storage_path),
            embedding_dimension=self._settings.embedding_dimension,
            distance=self._settings.distance.name.lower(),
        )
