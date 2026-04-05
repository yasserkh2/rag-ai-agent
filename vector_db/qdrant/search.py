from __future__ import annotations

from vector_db.contracts import VectorSearcher
from vector_db.models import VectorSearchMatch
from vector_db.qdrant.setup import QdrantSettings, QdrantVectorDatabaseSetup


class QdrantVectorSearcher(VectorSearcher):
    def __init__(
        self,
        settings: QdrantSettings | None = None,
        setup: QdrantVectorDatabaseSetup | None = None,
    ) -> None:
        if setup is None and settings is None:
            raise ValueError("Provide either Qdrant settings or a setup instance.")

        if setup is not None:
            self._setup = setup
        else:
            self._setup = QdrantVectorDatabaseSetup(settings)

        self._client = self._setup.create_client()

    @property
    def settings(self) -> QdrantSettings:
        return self._setup.settings

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        with_vectors: bool = False,
    ) -> list[VectorSearchMatch]:
        response = self._client.query_points(
            collection_name=self.settings.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )

        return [
            self._to_search_match(point=point, with_vectors=with_vectors)
            for point in response.points
        ]

    def _to_search_match(
        self,
        point: object,
        with_vectors: bool,
    ) -> VectorSearchMatch:
        point_id = str(point.id)
        payload = dict(point.payload or {})
        record_id = str(payload.get("record_id", point_id))
        vector = None

        if with_vectors:
            vector_data = point.vector
            if isinstance(vector_data, list):
                vector = vector_data
            elif vector_data is not None:
                vector = list(vector_data)

        return VectorSearchMatch(
            point_id=point_id,
            record_id=record_id,
            score=float(point.score),
            payload=payload,
            vector=vector,
        )
