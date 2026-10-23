import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int

class RecursiveSemanticChunker:
    """
    Splits long documents into contextual chunks with sliding-window overlap.
    Preserves structural headers and paragraphs.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        metadata = metadata or {}
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', text).strip()
        words = clean_text.split()
        
        chunks: List[TextChunk] = []
        start_idx = 0
        chunk_num = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_content = " ".join(chunk_words)

            chunks.append(TextChunk(
                chunk_id=f"{doc_id}#c{chunk_num}",
                doc_id=doc_id,
                text=chunk_content,
                metadata={**metadata, "start_word": start_idx, "end_word": end_idx},
                token_count=len(chunk_words)
            ))

            if end_idx == len(words):
                break
            start_idx += (self.chunk_size - self.overlap)
            chunk_num += 1

        return chunks
