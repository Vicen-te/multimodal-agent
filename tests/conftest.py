"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from multimodal_agent.rag.chunking import chunk_document
from multimodal_agent.rag.retriever import HybridRetriever
from multimodal_agent.rag.store import HybridStore

from ._fakes import StubEmbedder


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def store(embedder: StubEmbedder) -> HybridStore:
    docs = {
        "langgraph": "LangGraph models agents as a graph with nodes and cycles.",
        "rrf": "Reciprocal rank fusion combines ranked lists using reciprocal ranks.",
        "embeddings": "Sentence embeddings map text to vectors for cosine similarity.",
    }
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_document(doc_id, text, parent_size=120, child_size=60))
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    built = HybridStore()
    built.add(chunks, embeddings)
    return built


@pytest.fixture
def retriever(store: HybridStore, embedder: StubEmbedder) -> HybridRetriever:
    return HybridRetriever(store, embedder, top_k=2, candidate_k=5)
