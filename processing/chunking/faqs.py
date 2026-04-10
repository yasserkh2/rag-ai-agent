from __future__ import annotations

from processing.chunking.contracts import ChunkingStrategy
from processing.chunking.models import ChunkingInput, TextChunk


class FaqChunkingStrategy(ChunkingStrategy):
    def chunk(self, item: ChunkingInput) -> list[TextChunk]:
        text = item.text
        metadata = item.metadata
        extras: list[str] = []
        category = str(metadata.get("category", "")).strip()
        if category:
            extras.append(f"Category: {category}")
        difficulty = str(metadata.get("difficulty", "")).strip()
        if difficulty:
            extras.append(f"Difficulty: {difficulty}")
        if extras:
            text = f"{text}\n" + "\n".join(extras)
        return [
            TextChunk(
                chunk_id=f"{item.record_id}_chunk_0001",
                text=text,
                metadata=metadata,
            )
        ]
