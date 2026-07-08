"""Hybrid retrieval: fuse semantic and lexical rankings with RRF."""

from __future__ import annotations

from typing import Protocol

from .chunking import Chunk
from .store import HybridStore


class Embedder(Protocol):
    def embed(self, texts: list[str]): ...


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Combine ranked id lists into one ranking.

    Each item scores ``1 / (k + rank)`` in every list it appears in, summed across
    lists. RRF needs no score calibration between rankers, which is why it beats
    naive score addition when fusing BM25 with cosine similarity.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Retrieve parent chunks via fused semantic + lexical search."""

    def __init__(
        self,
        store: HybridStore,
        embedder: Embedder,
        top_k: int = 4,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str) -> list[Chunk]:
        if len(self.store) == 0:
            return []

        query_embedding = self.embedder.embed([query])[0]
        semantic = self.store.semantic_search(query_embedding, self.candidate_k)
        lexical = self.store.lexical_search(query, self.candidate_k)

        fused = reciprocal_rank_fusion(
            [[i for i, _ in semantic], [i for i, _ in lexical]], self.rrf_k
        )

        results: list[Chunk] = []
        seen_parents: set[str] = set()
        for index, _ in fused:
            chunk = self.store.chunks[index]
            if chunk.parent_id in seen_parents:
                continue
            seen_parents.add(chunk.parent_id)
            results.append(chunk)
            if len(results) >= self.top_k:
                break
        return results
