from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from processing.chunking.models import ChunkingInput
from processing.ingestion_pipeline.contracts import IngestionPipeline
from processing.ingestion_pipeline.models import IngestionResult, IngestionSource


@dataclass(frozen=True, slots=True)
class ProcessedDocumentRecord:
    doc_id: str
    service_id: str
    service_name: str
    title: str
    source_type: str
    source_file: str
    markdown_text: str

    @property
    def record_id(self) -> str:
        return self.doc_id

    @property
    def text(self) -> str:
        return self.markdown_text

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "service_id": self.service_id,
            "service_name": self.service_name,
            "title": self.title,
            "source_type": self.source_type,
            "source_file": self.source_file,
        }

    def as_chunking_input(self) -> ChunkingInput:
        return ChunkingInput(
            record_id=self.record_id,
            text=self.text,
            metadata=self.metadata,
        )


class DocumentManifestIngestionPipeline(IngestionPipeline):
    def __init__(self) -> None:
        self._processed_records: list[ProcessedDocumentRecord] = []

    @property
    def processed_records(self) -> list[ProcessedDocumentRecord]:
        return list(self._processed_records)

    def ingest(self, source: IngestionSource) -> IngestionResult:
        manifest_path = Path(source.file_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Document manifest must be a JSON array.")

        records = [
            self._build_record(
                payload=item,
                manifest_path=manifest_path,
                index=index,
            )
            for index, item in enumerate(payload, start=1)
        ]

        self._processed_records = records
        return IngestionResult(
            source_name=source.source_name,
            file_path=source.file_path,
            content_type=source.content_type,
            records_processed=len(records),
            points_upserted=0,
        )

    def _build_record(
        self,
        payload: object,
        manifest_path: Path,
        index: int,
    ) -> ProcessedDocumentRecord:
        if not isinstance(payload, dict):
            raise ValueError(f"Document manifest item {index} must be an object.")

        required_fields = (
            "doc_id",
            "service_id",
            "service_name",
            "title",
            "file_path",
            "source_type",
        )
        missing_fields = [
            field_name
            for field_name in required_fields
            if not isinstance(payload.get(field_name), str)
            or not str(payload[field_name]).strip()
        ]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"Document manifest item {index} is missing required fields: {missing}"
            )

        source_type = str(payload["source_type"]).strip()
        if source_type != "document":
            raise ValueError(
                f"Document manifest item {index} has unsupported source_type "
                f"'{source_type}'. Expected 'document'."
            )

        document_path = manifest_path.parent / str(payload["file_path"]).strip()
        markdown_text = document_path.read_text(encoding="utf-8").strip()
        if not markdown_text:
            raise ValueError(
                f"Document manifest item {index} references an empty markdown file: "
                f"{document_path.name}"
            )

        return ProcessedDocumentRecord(
            doc_id=str(payload["doc_id"]).strip(),
            service_id=str(payload["service_id"]).strip(),
            service_name=str(payload["service_name"]).strip(),
            title=str(payload["title"]).strip(),
            source_type=source_type,
            source_file=document_path.name,
            markdown_text=markdown_text,
        )
