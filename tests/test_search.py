import pytest
from fastapi.testclient import TestClient

from src.core.chunker import SlidingWindowChunker
from src.core.engine import HybridSearchEngine
from src.evals.benchmark import RetrievalEvaluationRunner
from src.main import app


def test_chunker_validates_configuration_and_preserves_overlap():
    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=10, overlap=10)

    chunks = SlidingWindowChunker(chunk_size=5, overlap=2).chunk_document(
        "doc", "one two three four five six seven eight"
    )
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_hybrid_search_returns_relevant_document_first():
    engine = HybridSearchEngine()
    engine.index_document(
        "security", "API credentials are protected with AES-256 encryption."
    )
    engine.index_document(
        "billing", "Monthly invoices are generated on the first calendar day."
    )

    response = engine.search("Which encryption protects API credentials?", top_k=2)

    assert response.results[0].doc_id == "security"
    assert "AES-256" in response.results[0].snippet


def test_search_does_not_return_zero_score_documents():
    engine = HybridSearchEngine()
    engine.index_document("billing", "Invoices are sent monthly.")

    response = engine.search("quantum telescope", mode="bm25")

    assert response.results == []


def test_duplicate_document_is_rejected():
    engine = HybridSearchEngine()
    engine.index_document("one", "First version")
    with pytest.raises(ValueError, match="already indexed"):
        engine.index_document("one", "Second version")


def test_synthetic_evaluation_reports_retrieval_metrics():
    result = RetrievalEvaluationRunner().run(top_k=2)
    assert result["dataset"] == "synthetic"
    assert result["metrics"]["hybrid"]["recall_at_2"] >= 0.75
    assert result["metrics"]["hybrid"]["mean_reciprocal_rank"] > 0


def test_api_validates_inputs_and_rejects_duplicate_ids():
    client = TestClient(app)
    document = {
        "doc_id": "api_limits_test",
        "content": "HTTP 429 indicates that a request has been rate limited.",
    }
    assert client.post("/api/v1/documents", json=document).status_code == 200
    assert client.post("/api/v1/documents", json=document).status_code == 409

    search = client.post(
        "/api/v1/search", json={"query": "rate limit status", "top_k": 0}
    )
    assert search.status_code == 422
