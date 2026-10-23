import math
import hashlib
from typing import List, Tuple
from .chunker import TextChunk

def generate_mock_embedding(text: str, dimensions: int = 64) -> List[float]:
    """
    Deterministic pseudo-embedding for testing and standalone execution
    without requiring external paid API keys.
    """
    vector = [0.0] * dimensions
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for i in range(dimensions):
            vector[i] += ((h >> (i % 32)) & 0xFF) / 255.0 - 0.5
            
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [round(x / norm, 4) for x in vector]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class DenseVectorStore:
    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions
        self.index: List[Tuple[TextChunk, List[float]]] = []

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        for chunk in chunks:
            emb = generate_mock_embedding(chunk.text, self.dimensions)
            self.index.append((chunk, emb))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        query_vec = generate_mock_embedding(query, self.dimensions)
        scored = []
        for chunk, emb in self.index:
            score = cosine_similarity(query_vec, emb)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
