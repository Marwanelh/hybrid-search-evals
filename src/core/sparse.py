import math
import re
from collections import Counter

from .chunker import TextChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: list[TextChunk] = []
        self.doc_lengths: list[int] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        for chunk in chunks:
            terms = tokenize(chunk.text)
            frequencies = Counter(terms)
            self.corpus.append(chunk)
            self.doc_lengths.append(len(terms))
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())

    def search(self, query: str, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        if not self.corpus:
            return []
        query_terms = tokenize(query)
        average_length = sum(self.doc_lengths) / len(self.doc_lengths)
        total_documents = len(self.corpus)
        scored = []

        for index, chunk in enumerate(self.corpus):
            score = 0.0
            frequencies = self.term_frequencies[index]
            for term in query_terms:
                term_count = frequencies.get(term, 0)
                if not term_count:
                    continue
                document_frequency = self.document_frequencies[term]
                inverse_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = self.doc_lengths[index] / average_length if average_length else 0
                denominator = term_count + self.k1 * (1 - self.b + self.b * length_ratio)
                score += inverse_frequency * term_count * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((chunk, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
