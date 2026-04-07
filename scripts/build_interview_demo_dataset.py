from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "cob_mock_kb_large"
OUTPUT_ROOT = SOURCE_ROOT / "interview_demo_kb"

HIGH_QUALITY_DOCUMENTS_ROOT = SOURCE_ROOT / "high_quality_documents"
HIGH_QUALITY_FAQS_ROOT = SOURCE_ROOT / "high_quality_faqs"
VERY_LARGE_MIXED_ROOT = SOURCE_ROOT / "very_large_mixed_kb"


def _reset_output_root() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_retrieval_corpus() -> None:
    retrieval_documents_root = OUTPUT_ROOT / "retrieval" / "documents"
    retrieval_faqs_root = OUTPUT_ROOT / "retrieval" / "faqs"

    manifest_source = HIGH_QUALITY_DOCUMENTS_ROOT / "documents_manifest.json"
    manifest_payload = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("Expected high-quality documents manifest to be a JSON array.")

    for item in manifest_payload:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_path", "")).strip()
        if not file_name:
            continue
        _copy_file(
            HIGH_QUALITY_DOCUMENTS_ROOT / file_name,
            retrieval_documents_root / file_name,
        )

    _copy_file(manifest_source, retrieval_documents_root / "documents_manifest.json")
    _copy_file(
        HIGH_QUALITY_FAQS_ROOT / "high_quality_faqs.jsonl",
        retrieval_faqs_root / "faqs.jsonl",
    )


def _copy_operations_corpus() -> None:
    operations_root = OUTPUT_ROOT / "operations"
    for relative_path in (
        Path("appointments/appointments.csv"),
        Path("appointments/appointments.jsonl"),
        Path("case_notes/case_notes.jsonl"),
        Path("structured/documents_index.json"),
        Path("structured/service_kpis.csv"),
        Path("structured/services.csv"),
        Path("structured/structured_data.json"),
    ):
        _copy_file(
            VERY_LARGE_MIXED_ROOT / relative_path,
            operations_root / relative_path,
        )


def _write_readme() -> None:
    readme = """# Interview Demo KB

This dataset is the interview-ready knowledge base layout for the demo chatbot.

## Design Goal

Keep the retrieval corpus clean:

- `retrieval/` is the only folder intended for RAG indexing
- `operations/` contains supporting operational data and should not be indexed for normal FAQ/document retrieval

That separation avoids the major overlap problem where rewritten high-quality retrieval files and the large mixed KB both describe the same services.

## Folder Map

- `retrieval/documents/`
  High-quality long-form documents for grounding.
- `retrieval/faqs/faqs.jsonl`
  High-quality concise FAQ set for direct answers.
- `operations/appointments/`
  Mock appointment records for booking flows.
- `operations/case_notes/`
  Mock case history for escalation flows.
- `operations/structured/`
  Canonical metadata for routing, normalization, and deterministic lookups.

## Recommended Usage

- Use `retrieval/` for vector indexing.
- Use `operations/structured/` for normalization and service metadata.
- Use `operations/appointments/` and `operations/case_notes/` only for workflow features, not as general RAG knowledge sources.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def _write_manifest() -> None:
    manifest = {
        "dataset_name": "Interview Demo KB",
        "version": "1.0",
        "generated_from": {
            "retrieval_documents": "cob_mock_kb_large/high_quality_documents",
            "retrieval_faqs": "cob_mock_kb_large/high_quality_faqs/high_quality_faqs.jsonl",
            "operations": "cob_mock_kb_large/very_large_mixed_kb",
        },
        "principles": [
            "retrieval and operational data are separated",
            "no large mixed documents or large mixed faqs are included in retrieval",
            "high-quality rewritten retrieval sources are preferred for demo RAG",
        ],
        "counts": {
            "retrieval_documents": 8,
            "retrieval_faqs": 48,
            "operations_files": 7,
        },
    }
    (OUTPUT_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    _reset_output_root()
    _copy_retrieval_corpus()
    _copy_operations_corpus()
    _write_readme()
    _write_manifest()
    print(f"Interview demo dataset generated at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
