import hashlib
import math
import re

from .chunker import TextChunk


def _features(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]


def hashed_feature_vector(text: str, dimensions: int = 256) -> list[float]:
    """Create a deterministic local feature vector; this is not a learned embedding."""
    vector = [0.0] * dimensions
    for feature in _features(text):
        digest = hashlib.sha256(feature.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class HashedFeatureIndex:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.index: list[tuple[TextChunk, list[float]]] = []

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        self.index.extend(
            (chunk, hashed_feature_vector(chunk.text, self.dimensions))
            for chunk in chunks
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        query_vector = hashed_feature_vector(query, self.dimensions)
        scored = [
            (chunk, cosine_similarity(query_vector, vector))
            for chunk, vector in self.index
        ]
        return sorted(
            ((chunk, score) for chunk, score in scored if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
