from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

T = TypeVar("T")

from app.config import load_runtime_config
from processing.chunking import DocumentChunkingStrategy
from processing.ingestion_pipeline import (
    DocumentManifestIngestionPipeline,
    IngestionSource,
)
from processing.vectorization import (
    DocumentVectorizationStrategy,
    build_embedding_generator,
)
from vector_db.contracts import VectorDatabaseSetup, VectorStore
from vector_db.qdrant import (
    QdrantSettings,
    QdrantVectorDatabaseSetup,
    QdrantVectorStore,
)


def _resolve_manifest_path() -> str:
    explicit_manifest_path = os.getenv("DOCUMENTS_MANIFEST_PATH", "").strip()
    if explicit_manifest_path:
        return explicit_manifest_path

    root_path = os.getenv("DOCUMENTS_ROOT_PATH", "").strip()
    if root_path:
        return str(Path(root_path) / "documents_manifest.json")

    return "cob_mock_kb_large/interview_demo_kb/retrieval/documents/documents_manifest.json"


def _build_source() -> IngestionSource:
    return IngestionSource(
        source_name="document_manifest",
        file_path=_resolve_manifest_path(),
        content_type="application/json",
    )


def _parse_limit() -> int | None:
    raw_limit = os.getenv("DOCUMENT_PIPELINE_LIMIT", "").strip()
    if not raw_limit:
        return None

    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError("DOCUMENT_PIPELINE_LIMIT must be greater than zero.")
    return limit


def _parse_batch_size() -> int:
    raw_batch_size = os.getenv("DOCUMENT_PIPELINE_BATCH_SIZE", "16").strip()
    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        raise ValueError("DOCUMENT_PIPELINE_BATCH_SIZE must be greater than zero.")
    return batch_size


def _batched(items: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _print_progress(
    completed_batches: int,
    total_batches: int,
    processed_records: int,
    total_records: int,
) -> None:
    bar_width = 30
    progress_ratio = completed_batches / total_batches if total_batches else 1.0
    filled_width = math.floor(progress_ratio * bar_width)
    bar = "#" * filled_width + "-" * (bar_width - filled_width)
    percent = progress_ratio * 100
    print(
        f"[{bar}] {percent:5.1f}% "
        f"({completed_batches}/{total_batches} batches, "
        f"{processed_records}/{total_records} records)",
        flush=True,
    )


def main() -> None:
    load_runtime_config(
        config_path=PROJECT_ROOT / "config.yml",
        env_path=PROJECT_ROOT / ".env",
    )

    source = _build_source()
    limit = _parse_limit()
    batch_size = _parse_batch_size()
    qdrant_settings = QdrantSettings.from_env(
        collection_env_key="QDRANT_DOCUMENT_COLLECTION",
        collection_default="customer_care_documents_kb",
    )

    ingestion_pipeline = DocumentManifestIngestionPipeline()
    chunking_strategy = DocumentChunkingStrategy()
    vectorization_strategy = DocumentVectorizationStrategy(
        build_embedding_generator(qdrant_settings.embedding_dimension)
    )
    qdrant_setup = QdrantVectorDatabaseSetup(qdrant_settings)
    vector_database: VectorDatabaseSetup = qdrant_setup
    vector_store: VectorStore = QdrantVectorStore(setup=qdrant_setup)

    setup_result = vector_database.ensure_collection()

    ingestion_result = ingestion_pipeline.ingest(source)
    processed_records = ingestion_pipeline.processed_records
    if limit is not None:
        processed_records = processed_records[:limit]

    record_batches = _batched(processed_records, batch_size)
    total_batches = len(record_batches)
    total_chunks = 0
    total_vector_records = 0
    total_upserted = 0
    sample_record = None

    if total_batches:
        print(
            f"Starting document processing for {len(processed_records)} records "
            f"across {total_batches} batch(es)...",
            flush=True,
        )

    processed_record_count = 0
    for batch_index, record_batch in enumerate(record_batches, start=1):
        batch_start_record = processed_record_count + 1
        batch_end_record = processed_record_count + len(record_batch)
        print(
            f"Starting batch {batch_index}/{total_batches} "
            f"(records {batch_start_record}-{batch_end_record})...",
            flush=True,
        )
        chunks = []
        for record in record_batch:
            chunks.extend(chunking_strategy.chunk(record.as_chunking_input()))

        vectorization_result = vectorization_strategy.vectorize(chunks)
        upsert_result = vector_store.upsert_records(vectorization_result.vector_records)

        total_chunks += len(chunks)
        total_vector_records += vectorization_result.records_processed
        total_upserted += upsert_result.points_upserted
        processed_record_count += len(record_batch)

        if sample_record is None and vectorization_result.vector_records:
            sample_record = vectorization_result.vector_records[0]

        _print_progress(
            completed_batches=batch_index,
            total_batches=total_batches,
            processed_records=processed_record_count,
            total_records=len(processed_records),
        )

    print("Document processing pipeline complete.")
    print(f"Source file: {source.file_path}")
    print(f"Collection: {setup_result.collection_name}")
    print(f"Embedding dimension: {qdrant_settings.embedding_dimension}")
    print(f"Records ingested: {ingestion_result.records_processed}")
    print(f"Records selected: {len(processed_records)}")
    print(f"Batch size: {batch_size}")
    print(f"Chunks created: {total_chunks}")
    print(f"Vector records created: {total_vector_records}")
    print(f"Points upserted: {total_upserted}")

    if sample_record is not None:
        print("\nSample vector record:")
        print(f"Record ID: {sample_record.record_id}")
        print(f"Text preview: {sample_record.text[:160]}")
        print(f"Embedding dimension: {len(sample_record.embedding)}")
        print(
            "Metadata: "
            + json.dumps(sample_record.metadata, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
