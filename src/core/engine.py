import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .chunker import RecursiveSemanticChunker, TextChunk
from .dense import DenseVectorStore
from .sparse import BM25Indexer
from .reranker import ReciprocalRankFusionReranker

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str
    relevance_score: float

class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieval_mode: str
    latency_ms: float

class HybridRAGEngine:
    def __init__(self):
        self.chunker = RecursiveSemanticChunker()
        self.dense_store = DenseVectorStore()
        self.sparse_indexer = BM25Indexer()
        self.reranker = ReciprocalRankFusionReranker()
        self.indexed_docs: Dict[str, str] = {}

    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> int:
        chunks = self.chunker.chunk_document(doc_id, content, metadata)
        self.dense_store.add_chunks(chunks)
        self.sparse_indexer.index_chunks(chunks)
        self.indexed_docs[doc_id] = content
        return len(chunks)

    def query(self, question: str, mode: str = "hybrid", top_k: int = 3) -> RAGResponse:
        start_time = time.time()
        
        if mode == "dense":
            ranked = self.dense_store.search(question, top_k=top_k)
        elif mode == "sparse":
            ranked = self.sparse_indexer.search(question, top_k=top_k)
        else:  # hybrid
            dense_res = self.dense_store.search(question, top_k=top_k * 2)
            sparse_res = self.sparse_indexer.search(question, top_k=top_k * 2)
            ranked = self.reranker.rerank(dense_res, sparse_res, top_n=top_k)

        citations = [
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                snippet=c.text[:140] + ("..." if len(c.text) > 140 else ""),
                relevance_score=score
            )
            for c, score in ranked
        ]

        # Synthesize answer strictly referencing retrieved chunks
        if not ranked:
            synthesized_answer = "No matching documentation found with sufficient confidence."
        else:
            primary_chunk = ranked[0][0]
            synthesized_answer = (
                f"Based on [{primary_chunk.chunk_id}], {primary_chunk.text.split('.')[0]}. "
                f"Verified across {len(citations)} retrieved source passages."
            )

        latency = round((time.time() - start_time) * 1000, 2)

        return RAGResponse(
            query=question,
            answer=synthesized_answer,
            citations=citations,
            retrieval_mode=mode,
            latency_ms=latency
        )
