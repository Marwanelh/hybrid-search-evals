from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core.engine import HybridSearchEngine, SearchResponse
from .evals.benchmark import RetrievalEvaluationRunner


app = FastAPI(
    title="Hybrid Search and Evaluation Lab",
    description="A local reference implementation of BM25, hashed feature retrieval, reciprocal-rank fusion, and retrieval evaluation.",
    version="2.0.0",
)
engine = HybridSearchEngine()


class DocumentInput(BaseModel):
    doc_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    content: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    mode: Literal["features", "bm25", "hybrid"] = "hybrid"
    top_k: int = Field(default=3, ge=1, le=20)


@app.get("/health", tags=["monitoring"])
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "hybrid-search-evals"}


@app.post("/api/v1/documents", tags=["index"])
async def index_document(document: DocumentInput) -> dict[str, Any]:
    try:
        chunks = engine.index_document(
            document.doc_id, document.content, document.metadata
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"status": "indexed", "doc_id": document.doc_id, "chunks": chunks}


@app.post("/api/v1/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchRequest) -> SearchResponse:
    return engine.search(request.query, request.mode, request.top_k)


@app.get("/api/v1/evaluations", tags=["evaluation"])
async def run_evaluations() -> dict[str, Any]:
    return RetrievalEvaluationRunner().run()


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001)
