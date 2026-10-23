import math
import re
from typing import List, Dict, Tuple
from collections import Counter
from .chunker import TextChunk

class BM25Indexer:
    """
    Standard BM25 keyword retrieval engine (Robertson et al.).
    Handles exact lexical matching, acronyms, and part numbers that dense vectors miss.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[TextChunk] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        self.corpus.extend(chunks)
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        query_terms = self._tokenize(query)
        scores = []
        N = len(self.corpus)
        if N == 0:
            return []

        for idx, chunk in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf = self.term_freqs[idx]

            for term in query_terms:
                if term in tf:
                    df = self.doc_freqs.get(term, 0)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    term_count = tf[term]
                    numerator = term_count * (self.k1 + 1)
                    denominator = term_count + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)

            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
