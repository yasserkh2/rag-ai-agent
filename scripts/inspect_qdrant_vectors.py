from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_env_file
from vector_db.qdrant import QdrantSettings
from vector_db.record_management import QdrantVectorRecordReader


def _parse_limit() -> int:
    raw_limit = os.getenv("VECTOR_INSPECT_LIMIT", "3").strip()
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError("VECTOR_INSPECT_LIMIT must be greater than zero.")
    return limit


def _parse_with_vectors() -> bool:
    return os.getenv("VECTOR_INSPECT_WITH_VECTORS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")

    settings = QdrantSettings.from_env()
    reader = QdrantVectorRecordReader(settings=settings)

    limit = _parse_limit()
    with_vectors = _parse_with_vectors()
    records = reader.list_records(limit=limit, with_vectors=with_vectors)

    print(f"Collection: {settings.collection_name}")
    print(f"Backend: {settings.url or settings.storage_path}")
    print(f"Stored points: {reader.count_records()}")
    print(f"Showing: {len(records)}")

    for index, record in enumerate(records, start=1):
        print(f"\nRecord {index}")
        print(f"Point ID: {record.point_id}")
        print(f"Record ID: {record.record_id}")
        print("Payload:")
        print(json.dumps(record.payload, indent=2, sort_keys=True))

        if record.vector is not None:
            print(f"Vector dimension: {len(record.vector)}")
            print(f"First 10 values: {record.vector[:10]}")


if __name__ == "__main__":
    main()
