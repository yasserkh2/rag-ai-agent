from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from processing.chunking.contracts import ChunkingStrategy
from processing.chunking.models import ChunkingInput, TextChunk

_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_SECTION_PATTERN = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class _DocumentSection:
    title: str
    body: str


class DocumentChunkingStrategy(ChunkingStrategy):
    def chunk(self, item: ChunkingInput) -> list[TextChunk]:
        document_title = self._extract_document_title(item.text) or str(
            item.metadata.get("title", "")
        ).strip()
        sections = self._extract_sections(item.text)
        if not sections:
            return [
                TextChunk(
                    chunk_id=f"{item.record_id}_chunk_0001",
                    text=self._build_chunk_text(
                        title=document_title,
                        service_name=str(item.metadata.get("service_name", "")).strip(),
                        section_title="Document",
                        section_blocks=[("Document", item.text.strip())],
                    ),
                    metadata=dict(item.metadata),
                )
            ]

        section_map = {section.title: section.body for section in sections}
        example_questions = section_map.get("Example Customer Questions", "")
        keywords = section_map.get("Keywords", "")

        chunk_specs: list[tuple[str, list[tuple[str, str]]]] = []

        overview_blocks = self._non_empty_blocks(
            [
                ("Service Overview", section_map.get("Service Overview", "")),
                (
                    "What This Service Usually Includes",
                    section_map.get("What This Service Usually Includes", ""),
                ),
            ]
        )
        if overview_blocks:
            chunk_specs.append(
                ("Service Overview | What This Service Usually Includes", overview_blocks)
            )

        for title in (
            "Common Practice Scenarios",
            "Information To Collect During Intake",
            "How The Chatbot Should Respond",
            "Escalation Guidance",
        ):
            body = section_map.get(title, "")
            if not body:
                continue

            blocks = [(title, body)]
            if title == "How The Chatbot Should Respond" and example_questions:
                blocks.append(("Example Customer Questions", example_questions))
            chunk_specs.append((title, blocks))

        chunks: list[TextChunk] = []
        for index, (section_title, blocks) in enumerate(chunk_specs, start=1):
            metadata = dict(item.metadata)
            metadata["section_title"] = section_title
            if keywords:
                metadata["keywords"] = keywords

            chunks.append(
                TextChunk(
                    chunk_id=f"{item.record_id}_chunk_{index:04d}",
                    text=self._build_chunk_text(
                        title=document_title,
                        service_name=str(metadata.get("service_name", "")).strip(),
                        section_title=section_title,
                        section_blocks=blocks,
                    ),
                    metadata=metadata,
                )
            )

        return chunks

    def _extract_document_title(self, text: str) -> str:
        match = _TITLE_PATTERN.search(text)
        if match is None:
            return ""
        return match.group("title").strip()

    def _extract_sections(self, text: str) -> list[_DocumentSection]:
        matches = list(_SECTION_PATTERN.finditer(text))
        if not matches:
            return []

        sections: list[_DocumentSection] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            sections.append(
                _DocumentSection(
                    title=match.group("title").strip(),
                    body=body,
                )
            )
        return sections

    def _non_empty_blocks(
        self,
        blocks: Iterable[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        return [(title, body.strip()) for title, body in blocks if body.strip()]

    def _build_chunk_text(
        self,
        *,
        title: str,
        service_name: str,
        section_title: str,
        section_blocks: list[tuple[str, str]],
    ) -> str:
        lines = [
            f"Title: {title}",
            f"Service: {service_name}",
            f"Section: {section_title}",
            "",
        ]
        for index, (block_title, block_body) in enumerate(section_blocks):
            if index > 0:
                lines.append("")
            lines.append(block_title)
            lines.append(block_body)
        return "\n".join(lines).strip()
