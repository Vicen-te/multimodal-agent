"""Load the markdown corpus, chunk it, embed it, and build a hybrid store."""

from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .retriever import Embedder
from .store import HybridStore

CORPUS_DIR = Path(__file__).parent / "corpus"


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[tuple[str, str]]:
    """Return ``(doc_id, text)`` pairs for every markdown file in the corpus."""
    documents: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        documents.append((path.stem, path.read_text(encoding="utf-8")))
    return documents


def build_store(embedder: Embedder, corpus_dir: Path = CORPUS_DIR) -> HybridStore:
    """Chunk and embed the corpus into a ready-to-query :class:`HybridStore`."""
    chunks = []
    for doc_id, text in load_corpus(corpus_dir):
        chunks.extend(chunk_document(doc_id, text))

    embeddings = embedder.embed([chunk.text for chunk in chunks])
    store = HybridStore()
    store.add(chunks, embeddings)
    return store
