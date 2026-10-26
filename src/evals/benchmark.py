import json
from pathlib import Path
from typing import Any

from ..core.engine import HybridSearchEngine


class RetrievalEvaluationRunner:
    def __init__(self, dataset_path: str | None = None) -> None:
        path = Path(dataset_path) if dataset_path else Path(__file__).with_name("dataset.json")
        self.cases: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
        if not self.cases:
            raise ValueError("evaluation dataset must not be empty")

    def run(self, top_k: int = 2) -> dict[str, Any]:
        engine = HybridSearchEngine()
        for case in self.cases:
            engine.index_document(case["doc_id"], case["document"])

        metrics = {}
        for mode in ("features", "bm25", "hybrid"):
            reciprocal_ranks = []
            hits = 0
            for case in self.cases:
                response = engine.search(case["question"], mode=mode, top_k=top_k)
                ids = [result.doc_id for result in response.results]
                if case["doc_id"] in ids:
                    rank = ids.index(case["doc_id"]) + 1
                    hits += 1
                    reciprocal_ranks.append(1 / rank)
                else:
                    reciprocal_ranks.append(0)
            metrics[mode] = {
                f"recall_at_{top_k}": round(hits / len(self.cases), 3),
                "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(self.cases), 3),
                "queries": len(self.cases),
            }
        return {"dataset": "synthetic", "top_k": top_k, "metrics": metrics}
