"""In-memory hybrid store: dense vectors for semantics, BM25 for lexical match."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridStore:
    """Holds chunks plus their embeddings and a BM25 index over child text."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = list(chunks)
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def semantic_search(self, query_embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._embeddings is None or len(self.chunks) == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-12
        scores = (self._embeddings @ query) / denom
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def lexical_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None or len(self.chunks) == 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]
