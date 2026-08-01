"""Lightweight test doubles so the agent and RAG logic can run fully offline."""

from __future__ import annotations

import numpy as np

from multimodal_agent.rag.chunking import Chunk


class FakeChat:
    """Replays a scripted list of AIMessages across invoke / bind_tools calls."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._index = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self._responses[self._index]
        self._index += 1
        return response


class StubEmbedder:
    """Deterministic hashing embedder: no model download, stable vectors."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                vectors[row, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class StubVision:
    def __init__(self, reply: str = "stub image description") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def describe(self, image_base64: str, query: str) -> str:
        self.calls.append((image_base64, query))
        return self.reply


class StubRetriever:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or [
            Chunk("d::p0::c0", "d", "child text", "d::p0", "parent text")
        ]
        self.queries: list[str] = []

    def search(self, query: str) -> list[Chunk]:
        self.queries.append(query)
        return self.chunks
