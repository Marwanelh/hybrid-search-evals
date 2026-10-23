from typing import List, Dict, Tuple
from .chunker import TextChunk

class ReciprocalRankFusionReranker:
    """
    RRF (Reciprocal Rank Fusion) algorithm to merge multiple ranked retrieval lists
    without requiring score normalization across different scales.
    Score = sum(1 / (k + rank_i))
    """
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def rerank(
        self, 
        dense_results: List[Tuple[TextChunk, float]], 
        sparse_results: List[Tuple[TextChunk, float]], 
        top_n: int = 3
    ) -> List[Tuple[TextChunk, float]]:
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, TextChunk] = {}

        # Dense rank contributions
        for rank, (chunk, _) in enumerate(dense_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sparse rank contributions
        for rank, (chunk, _) in enumerate(sparse_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], round(score, 5)) for cid, score in sorted_chunks[:top_n]]
