import re
from typing import Any

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    word_count: int


class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 160, overlap: int = 30) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not doc_id.strip() or not clean_text:
            raise ValueError("doc_id and text must not be empty")

        words = clean_text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for chunk_number, start in enumerate(range(0, len(words), step)):
            selected = words[start : start + self.chunk_size]
            if not selected:
                break
            end = start + len(selected)
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}#c{chunk_number}",
                    doc_id=doc_id,
                    text=" ".join(selected),
                    metadata={**(metadata or {}), "start_word": start, "end_word": end},
                    word_count=len(selected),
                )
            )
            if end == len(words):
                break
        return chunks
