import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from .core.engine import HybridRAGEngine, RAGResponse
from .evals.benchmark import RAGEvaluationRunner

app = FastAPI(
    title="Production Hybrid RAG with Automated Evals",
    description="Enterprise document retrieval with dense vector similarity, BM25 keyword matching, cross-encoder reranking, and automated Ragas-style evals.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = HybridRAGEngine()

# Seed default architecture docs
engine.index_document(
    "arch_pgvector",
    "PostgreSQL with pgvector provides integrated transactional safety and HNSW vector similarity search for production RAG systems."
)
engine.index_document(
    "arch_hybrid",
    "Hybrid search fuses dense semantic embeddings with sparse BM25 lexical tokens using Reciprocal Rank Fusion (RRF) to eliminate out-of-vocabulary failures."
)

class DocumentIngest(BaseModel):
    doc_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"  # "dense", "sparse", or "hybrid"
    top_k: Optional[int] = 3

@app.get("/health", tags=["Monitoring"])
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "hybrid-rag-evals"}

@app.post("/api/v1/index", tags=["Ingestion"])
async def index_document(doc: DocumentIngest) -> Dict[str, Any]:
    chunks_created = engine.index_document(doc.doc_id, doc.content, doc.metadata)
    return {"status": "indexed", "doc_id": doc.doc_id, "chunks_created": chunks_created}

@app.post("/api/v1/query", response_model=RAGResponse, tags=["RAG Retrieval"])
async def query_knowledge_base(req: QueryRequest) -> RAGResponse:
    if req.mode not in ["dense", "sparse", "hybrid"]:
        raise HTTPException(status_code=400, detail="mode must be one of: dense, sparse, hybrid")
    return engine.query(req.query, mode=req.mode, top_k=req.top_k)

@app.get("/api/v1/evals/run", tags=["Automated Evaluations"])
async def run_evaluation_suite() -> Dict[str, Any]:
    runner = RAGEvaluationRunner()
    return runner.run_benchmark()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
