from .chunker import TextChunk


class ReciprocalRankFusion:
    def __init__(self, rank_constant: int = 60) -> None:
        self.rank_constant = rank_constant

    def merge(
        self,
        *ranked_lists: list[tuple[TextChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        scores: dict[str, float] = {}
        chunks: dict[str, TextChunk] = {}
        for ranked in ranked_lists:
            for rank, (chunk, _) in enumerate(ranked, start=1):
                chunks[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (
                    self.rank_constant + rank
                )
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(chunks[chunk_id], score) for chunk_id, score in ordered[:top_k]]
