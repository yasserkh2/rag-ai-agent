from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from processing.chunking.contracts import ChunkingStrategy
from processing.chunking.models import ChunkingInput, TextChunk

_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_SECTION_PATTERN = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_KEYWORD_SPLIT_PATTERN = re.compile(r"[,;\n]")
_KEYWORD_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")


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
                        doc_id=str(item.metadata.get("doc_id", "")).strip(),
                        source_file=str(item.metadata.get("source_file", "")).strip(),
                        title=document_title,
                        service_name=str(item.metadata.get("service_name", "")).strip(),
                        section_title="Document",
                        section_blocks=[("Document", item.text.strip())],
                        keywords=str(item.metadata.get("keywords", "")).strip(),
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
                        doc_id=str(metadata.get("doc_id", "")).strip(),
                        source_file=str(metadata.get("source_file", "")).strip(),
                        title=document_title,
                        service_name=str(metadata.get("service_name", "")).strip(),
                        section_title=section_title,
                        section_blocks=blocks,
                        keywords=str(metadata.get("keywords", "")).strip(),
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
        doc_id: str,
        source_file: str,
        title: str,
        service_name: str,
        section_title: str,
        section_blocks: list[tuple[str, str]],
        keywords: str = "",
    ) -> str:
        lines: list[str] = []
        if doc_id:
            lines.append(f"Document ID: {doc_id}")
        if source_file:
            lines.append(f"Document File: {source_file}")
        lines.extend(
            [
            f"Title: {title}",
            f"Service: {service_name}",
            f"Section: {section_title}",
            ]
        )
        if keywords:
            keyword_terms = self._normalize_keywords(keywords)
            if keyword_terms:
                lines.append(f"Keywords: {', '.join(keyword_terms)}")
                lines.extend(
                    self._build_keyword_hint_lines(
                        keyword_terms=keyword_terms,
                        service_name=service_name,
                        title=title,
                    )
                )
            else:
                lines.append(f"Keywords: {keywords}")
        lines.append("")
        for index, (block_title, block_body) in enumerate(section_blocks):
            if index > 0:
                lines.append("")
            lines.append(block_title)
            lines.append(block_body)
        return "\n".join(lines).strip()

    def _normalize_keywords(self, keywords: str) -> list[str]:
        normalized_terms: list[str] = []
        seen: set[str] = set()

        for raw_term in _KEYWORD_SPLIT_PATTERN.split(keywords):
            term = raw_term.strip().lstrip("-* ").strip()
            if not term:
                continue
            collapsed = " ".join(term.split())
            key = collapsed.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized_terms.append(collapsed)

        return normalized_terms

    def _build_keyword_hint_lines(
        self,
        *,
        keyword_terms: list[str],
        service_name: str,
        title: str,
    ) -> list[str]:
        lines: list[str] = []
        lines.append(f"Keyword Terms: {' | '.join(keyword_terms)}")

        keyword_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for term in keyword_terms:
            for token in _KEYWORD_TOKEN_PATTERN.findall(term.lower()):
                if len(token) < 4:
                    continue
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                keyword_tokens.append(token)
        if keyword_tokens:
            lines.append(f"Keyword Tokens: {' | '.join(keyword_tokens)}")

        raw_hints: list[str] = list(keyword_terms)
        for term in keyword_terms[:8]:
            if service_name:
                raw_hints.append(f"{service_name} {term}")
            if title and title.lower() != service_name.lower():
                raw_hints.append(f"{title} {term}")

        hint_terms: list[str] = []
        seen_hints: set[str] = set()
        for hint in raw_hints:
            normalized_hint = " ".join(hint.split())
            if not normalized_hint:
                continue
            hint_key = normalized_hint.lower()
            if hint_key in seen_hints:
                continue
            seen_hints.add(hint_key)
            hint_terms.append(normalized_hint)
            if len(hint_terms) >= 16:
                break

        if hint_terms:
            lines.append(f"Keyword Query Hints: {' ; '.join(hint_terms)}")

        return lines
