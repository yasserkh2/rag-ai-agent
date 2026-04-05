from __future__ import annotations

from app.config import load_env_file
from vector_db.contracts import VectorDatabaseSetup
from vector_db.qdrant.setup import QdrantSettings, QdrantVectorDatabaseSetup


def main() -> None:
    load_env_file()

    settings = QdrantSettings.from_env()
    vector_database: VectorDatabaseSetup = QdrantVectorDatabaseSetup(settings)
    result = vector_database.ensure_collection()

    status = "created" if result.created else "already exists"
    print("Qdrant setup complete.")
    print(f"Collection: {result.collection_name}")
    print(f"Status: {status}")
    print(f"Distance: {result.distance}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Backend: {result.backend}")


if __name__ == "__main__":
    main()
