from multimodal_agent.rag.chunking import chunk_document, split_text


def test_split_text_respects_size_and_word_boundaries():
    text = "one two three four five six seven eight nine ten"
    windows = split_text(text, size=15, overlap=0)
    assert len(windows) > 1
    for window in windows:
        assert window == window.strip()
        # never split inside a word
        for word in window.split():
            assert word in text.split()


def test_split_text_overlap_carries_context():
    text = " ".join(f"w{i}" for i in range(40))
    no_overlap = split_text(text, size=40, overlap=0)
    with_overlap = split_text(text, size=40, overlap=20)
    assert len(with_overlap) >= len(no_overlap)


def test_chunk_document_links_children_to_parents():
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_document("doc1", text, parent_size=200, child_size=80)
    assert chunks
    for chunk in chunks:
        assert chunk.doc_id == "doc1"
        assert chunk.parent_id.startswith("doc1::p")
        assert chunk.id.startswith(chunk.parent_id)
        assert chunk.text in chunk.parent_text


def test_chunk_document_empty_text():
    assert chunk_document("doc1", "") == []
