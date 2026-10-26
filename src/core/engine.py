from typing import Literal

from pydantic import BaseModel

from .chunker import SlidingWindowChunker
from .dense import HashedFeatureIndex
from .reranker import ReciprocalRankFusion
from .sparse import BM25Index


SearchMode = Literal["features", "bm25", "hybrid"]


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[SearchResult]


class HybridSearchEngine:
    def __init__(self) -> None:
        self.chunker = SlidingWindowChunker()
        self.feature_index = HashedFeatureIndex()
        self.bm25_index = BM25Index()
        self.fusion = ReciprocalRankFusion()
        self.document_ids: set[str] = set()

    def index_document(
        self, doc_id: str, content: str, metadata: dict | None = None
    ) -> int:
        if doc_id in self.document_ids:
            raise ValueError(f"document '{doc_id}' is already indexed")
        chunks = self.chunker.chunk_document(doc_id, content, metadata)
        self.feature_index.add_chunks(chunks)
        self.bm25_index.add_chunks(chunks)
        self.document_ids.add(doc_id)
        return len(chunks)

    def search(
        self, query: str, mode: SearchMode = "hybrid", top_k: int = 3
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        if mode == "features":
            ranked = self.feature_index.search(query, top_k)
        elif mode == "bm25":
            ranked = self.bm25_index.search(query, top_k)
        else:
            candidates = max(top_k * 3, 10)
            ranked = self.fusion.merge(
                self.feature_index.search(query, candidates),
                self.bm25_index.search(query, candidates),
                top_k=top_k,
            )

        return SearchResponse(
            query=query,
            mode=mode,
            results=[
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    snippet=chunk.text[:240],
                    score=round(score, 6),
                )
                for chunk, score in ranked
            ],
        )
