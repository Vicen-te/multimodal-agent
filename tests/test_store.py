import numpy as np
import pytest

from multimodal_agent.rag.chunking import Chunk
from multimodal_agent.rag.store import HybridStore, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, RRF-fusion!") == ["hello", "rrf", "fusion"]


def test_add_rejects_length_mismatch():
    store = HybridStore()
    chunk = Chunk("c0", "d", "text", "p0", "parent")
    with pytest.raises(ValueError):
        store.add([chunk], np.zeros((2, 4), dtype=np.float32))


def test_semantic_search_ranks_closest_vector_first():
    store = HybridStore()
    chunks = [Chunk(f"c{i}", "d", f"text {i}", f"p{i}", "parent") for i in range(3)]
    embeddings = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    store.add(chunks, embeddings)
    results = store.semantic_search(np.array([1, 0], dtype=np.float32), k=3)
    assert results[0][0] == 0


def test_lexical_search_finds_keyword():
    # A multi-doc corpus keeps BM25's IDF positive for a discriminating term;
    # with only two docs a term in one of them degenerates to IDF = log(1) = 0.
    store = HybridStore()
    texts = [
        "reciprocal rank fusion combines rankings",
        "sentence embeddings map text to vectors",
        "ollama runs language models locally",
        "gradio builds web interfaces quickly",
        "langgraph models agents as graphs",
    ]
    chunks = [Chunk(f"c{i}", "d", t, f"p{i}", "p") for i, t in enumerate(texts)]
    store.add(chunks, np.zeros((len(chunks), 2), dtype=np.float32))
    results = store.lexical_search("fusion", k=3)
    assert results[0][0] == 0


def test_empty_store_returns_no_results():
    store = HybridStore()
    assert store.semantic_search(np.zeros(2, dtype=np.float32), k=3) == []
    assert store.lexical_search("anything", k=3) == []
