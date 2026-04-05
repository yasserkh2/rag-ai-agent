from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_env_file
from processing.vectorization import build_embedding_generator
from vector_db.contracts import VectorSearcher
from vector_db.qdrant import QdrantSettings, QdrantVectorSearcher


def _parse_limit() -> int:
    raw_limit = os.getenv("FAQ_RETRIEVAL_LIMIT", "5").strip()
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError("FAQ_RETRIEVAL_LIMIT must be greater than zero.")
    return limit


def _parse_with_vectors() -> bool:
    return os.getenv("FAQ_RETRIEVAL_WITH_VECTORS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")

    query_text = os.getenv(
        "FAQ_RETRIEVAL_QUERY",
        "What does credentialing include?",
    ).strip()
    if not query_text:
        raise ValueError("FAQ_RETRIEVAL_QUERY must not be empty.")

    settings = QdrantSettings.from_env()
    embedding_generator = build_embedding_generator(settings.embedding_dimension)
    searcher: VectorSearcher = QdrantVectorSearcher(settings=settings)

    limit = _parse_limit()
    with_vectors = _parse_with_vectors()
    query_vector = embedding_generator.embed_query(query_text)
    matches = searcher.search(
        query_vector=query_vector,
        limit=limit,
        with_vectors=with_vectors,
    )

    print(f"Collection: {settings.collection_name}")
    print(f"Backend: {settings.url or settings.storage_path}")
    print(f"Query: {query_text}")
    print(f"Matches returned: {len(matches)}")

    for index, match in enumerate(matches, start=1):
        print(f"\nMatch {index}")
        print(f"Point ID: {match.point_id}")
        print(f"Record ID: {match.record_id}")
        print(f"Score: {match.score}")
        print("Payload:")
        print(json.dumps(match.payload, indent=2, sort_keys=True))

        if match.vector is not None:
            print(f"Vector dimension: {len(match.vector)}")
            print(f"First 10 values: {match.vector[:10]}")


if __name__ == "__main__":
    main()
