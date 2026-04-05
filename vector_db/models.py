from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VectorCollectionSetupResult:
    collection_name: str
    created: bool
    backend: str
    embedding_dimension: int
    distance: str


@dataclass(frozen=True, slots=True)
class VectorRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class VectorUpsertResult:
    collection_name: str
    points_upserted: int


@dataclass(frozen=True, slots=True)
class VectorSearchMatch:
    point_id: str
    record_id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None
