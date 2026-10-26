# Hybrid Search and Evaluation Lab

A local FastAPI reference implementation that combines BM25 lexical retrieval with deterministic hashed feature retrieval, merges rankings with Reciprocal Rank Fusion, and measures retrieval quality on a synthetic dataset.

## Why this project exists

Search systems need more than an attractive demo. They need explicit indexing rules, reproducible ranking, citations back to source chunks, validation at API boundaries, and evaluation data that can be rerun after a change.

This project focuses on those engineering fundamentals. It does **not** use learned embeddings, a vector database, an LLM, or client data, and it does not present local benchmark results as production performance.

## Architecture

```mermaid
flowchart LR
    A[Validated document] --> B[Sliding-window chunks]
    B --> C[BM25 index]
    B --> D[Hashed unigram + bigram feature index]
    C --> E[Reciprocal Rank Fusion]
    D --> E
    E --> F[Ranked source snippets]
    G[Synthetic evaluation cases] --> H[Recall@K and MRR]
    F --> H
```

## What it demonstrates

- Deterministic chunking with validated overlap
- BM25 retrieval implemented with the Python standard library
- Local hashed feature vectors for a second independent ranking signal
- Reciprocal Rank Fusion without incompatible score normalization
- Source-linked snippets instead of generated answers
- Duplicate-document protection and bounded request fields
- Recall@K and mean reciprocal rank evaluation
- FastAPI, Docker packaging, tests, and CI

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8001
```

Open `http://localhost:8001/docs` for the interactive API.

```bash
python -m pytest -q
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents` | Validate, chunk, and index one document |
| `POST` | `/api/v1/search` | Search with `features`, `bm25`, or `hybrid` mode |
| `GET` | `/api/v1/evaluations` | Run the bundled synthetic evaluation set |
| `GET` | `/health` | Health check |

Example:

```bash
curl -X POST http://localhost:8001/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"api_limits","content":"HTTP 429 indicates that a request has been rate limited."}'

curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Which status code means rate limited?","mode":"hybrid","top_k":3}'
```

## Tradeoffs

- The feature index is deterministic and dependency-free, but it is not a learned semantic embedding model.
- Indexes live in memory and reset on restart. Persistent storage and index versioning are required for deployment.
- Evaluation data is deliberately small and synthetic. A real system needs representative, reviewed queries and relevance labels.
- Returned snippets are retrieval evidence. Answer generation, faithfulness checks, access control, and tenant isolation are separate concerns.

## Scope

Independent portfolio project using synthetic examples only. It contains no FillFlow code, client documents, private prompts, credentials, or production metrics.

## Author

[Marwane Elhour](https://github.com/Marwanelh) · [LinkedIn](https://www.linkedin.com/in/marwanelh/)
