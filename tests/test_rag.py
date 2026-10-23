import pytest
from src.core.chunker import RecursiveSemanticChunker
from src.core.dense import DenseVectorStore
from src.core.sparse import BM25Indexer
from src.core.engine import HybridRAGEngine
from src.evals.benchmark import RAGEvaluationRunner

def test_chunker_sliding_window():
    chunker = RecursiveSemanticChunker(chunk_size=10, overlap=3)
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13"
    chunks = chunker.chunk_document("doc1", text)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc1#c0"

def test_hybrid_search_and_citations():
    engine = HybridRAGEngine()
    engine.index_document("doc_security", "All sensitive API credentials are encrypted with AES-256 in production.")
    engine.index_document("doc_billing", "Invoices are generated on the first day of each calendar month.")

    res = engine.query("What encryption standard protects credentials?", mode="hybrid")
    assert len(res.citations) > 0
    assert res.citations[0].doc_id == "doc_security"
    assert "AES-256" in res.citations[0].snippet

def test_eval_benchmark_runner():
    runner = RAGEvaluationRunner()
    benchmark = runner.run_benchmark()
    metrics = benchmark["evaluation_metrics"]
    assert "hybrid" in metrics
    assert metrics["hybrid"]["recall_at_2"] >= 0.6
