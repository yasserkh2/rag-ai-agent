from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from processing.chunking import DocumentChunkingStrategy
from processing.ingestion_pipeline import (
    DocumentManifestIngestionPipeline,
    IngestionSource,
)
from processing.vectorization import DocumentVectorizationStrategy


class StubEmbeddingGenerator:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts, start=1)]


class DocumentProcessingTests(unittest.TestCase):
    def test_manifest_ingestion_reads_document_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "doc_0001.md"
            markdown_path.write_text(_sample_document_markdown(), encoding="utf-8")
            manifest_path = temp_path / "documents_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "doc_0001",
                            "service_id": "svc_auth_benefits",
                            "service_name": "Authorizations and Benefits Verification",
                            "title": "Authorizations and Benefits Verification",
                            "file_path": markdown_path.name,
                            "source_type": "document",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = DocumentManifestIngestionPipeline()
            result = pipeline.ingest(
                IngestionSource(
                    source_name="document_manifest",
                    file_path=str(manifest_path),
                    content_type="application/json",
                )
            )

            self.assertEqual(result.records_processed, 1)
            record = pipeline.processed_records[0]
            self.assertEqual(record.doc_id, "doc_0001")
            self.assertEqual(record.service_id, "svc_auth_benefits")
            self.assertEqual(record.service_name, "Authorizations and Benefits Verification")
            self.assertEqual(record.metadata["source_type"], "document")
            self.assertEqual(record.source_file, "doc_0001.md")
            self.assertIn("## Service Overview", record.text)

    def test_manifest_ingestion_validates_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "doc_0001.md"
            markdown_path.write_text(_sample_document_markdown(), encoding="utf-8")
            manifest_path = temp_path / "documents_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "doc_0001",
                            "service_id": "svc_auth_benefits",
                            "service_name": "Authorizations and Benefits Verification",
                            "file_path": markdown_path.name,
                            "source_type": "document",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = DocumentManifestIngestionPipeline()
            with self.assertRaisesRegex(ValueError, "missing required fields: title"):
                pipeline.ingest(
                    IngestionSource(
                        source_name="document_manifest",
                        file_path=str(manifest_path),
                        content_type="application/json",
                    )
                )

    def test_chunking_splits_by_heading_and_merges_low_signal_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "doc_0001.md"
            markdown_path.write_text(_sample_document_markdown(), encoding="utf-8")
            manifest_path = temp_path / "documents_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "doc_0001",
                            "service_id": "svc_auth_benefits",
                            "service_name": "Authorizations and Benefits Verification",
                            "title": "Authorizations and Benefits Verification",
                            "file_path": markdown_path.name,
                            "source_type": "document",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = DocumentManifestIngestionPipeline()
            pipeline.ingest(
                IngestionSource(
                    source_name="document_manifest",
                    file_path=str(manifest_path),
                    content_type="application/json",
                )
            )
            record = pipeline.processed_records[0]
            chunks = DocumentChunkingStrategy().chunk(record.as_chunking_input())

            self.assertEqual(
                [chunk.chunk_id for chunk in chunks],
                [
                    "doc_0001_chunk_0001",
                    "doc_0001_chunk_0002",
                    "doc_0001_chunk_0003",
                    "doc_0001_chunk_0004",
                    "doc_0001_chunk_0005",
                ],
            )
            self.assertEqual(
                chunks[0].metadata["section_title"],
                "Service Overview | What This Service Usually Includes",
            )
            self.assertIn("What This Service Usually Includes", chunks[0].text)
            self.assertEqual(
                chunks[3].metadata["section_title"],
                "How The Chatbot Should Respond",
            )
            self.assertIn("Example Customer Questions", chunks[3].text)
            self.assertIn("keywords", chunks[4].metadata)
            self.assertNotIn("Section: Keywords", "\n\n".join(chunk.text for chunk in chunks))

    def test_vectorization_builds_document_vector_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "doc_0001.md"
            markdown_path.write_text(_sample_document_markdown(), encoding="utf-8")
            manifest_path = temp_path / "documents_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "doc_0001",
                            "service_id": "svc_auth_benefits",
                            "service_name": "Authorizations and Benefits Verification",
                            "title": "Authorizations and Benefits Verification",
                            "file_path": markdown_path.name,
                            "source_type": "document",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = DocumentManifestIngestionPipeline()
            pipeline.ingest(
                IngestionSource(
                    source_name="document_manifest",
                    file_path=str(manifest_path),
                    content_type="application/json",
                )
            )
            chunks = DocumentChunkingStrategy().chunk(
                pipeline.processed_records[0].as_chunking_input()
            )

            result = DocumentVectorizationStrategy(StubEmbeddingGenerator()).vectorize(chunks)

            self.assertEqual(result.records_processed, 5)
            record = result.vector_records[0]
            self.assertEqual(record.record_id, "doc_0001_chunk_0001")
            self.assertEqual(record.metadata["source_type"], "document")
            self.assertEqual(record.metadata["doc_id"], "doc_0001")
            self.assertEqual(
                record.metadata["section_title"],
                "Service Overview | What This Service Usually Includes",
            )
            self.assertEqual(record.embedding[0], 1.0)


def _sample_document_markdown() -> str:
    return textwrap.dedent(
        """
        # Authorizations and Benefits Verification

        ## Service Overview
        This service confirms coverage and helps prevent delays.

        ## What This Service Usually Includes
        - Eligibility checks
        - Benefits verification

        ## Common Practice Scenarios
        - Coverage is being checked manually.
        - Authorizations are slowing visits.

        ## Information To Collect During Intake
        - Practice name
        - Payer details

        ## How The Chatbot Should Respond
        Explain the service and collect the intake details.

        ## Escalation Guidance
        Escalate when case-specific insurance advice is requested.

        ## Example Customer Questions
        - What is included?
        - What should we prepare?

        ## Keywords
        prior authorization, benefits verification, insurance
        """
    ).strip()


if __name__ == "__main__":
    unittest.main()
