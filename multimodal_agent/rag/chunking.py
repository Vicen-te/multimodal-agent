"""Parent-child document chunking.

Retrieval matches small child chunks (precise) but the agent receives the larger
parent chunk (context). This keeps recall sharp without starving the model of
surrounding text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable child chunk that carries its parent context."""

    id: str
    doc_id: str
    text: str
    parent_id: str
    parent_text: str


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into word-aligned windows of about ``size`` characters."""
    words = text.split()
    if not words:
        return []

    windows: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > size:
            windows.append(" ".join(current))
            if overlap > 0:
                carried: list[str] = []
                carried_len = 0
                for prev in reversed(current):
                    carried_len += len(prev) + 1
                    carried.insert(0, prev)
                    if carried_len >= overlap:
                        break
                current = carried
                length = sum(len(w) + 1 for w in current)
            else:
                current = []
                length = 0
        current.append(word)
        length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        windows.append(" ".join(current))
    return windows


def chunk_document(
    doc_id: str,
    text: str,
    parent_size: int = 800,
    child_size: int = 250,
    overlap: int = 40,
) -> list[Chunk]:
    """Break ``text`` into parent windows, then child windows within each parent."""
    chunks: list[Chunk] = []
    for parent_index, parent_text in enumerate(split_text(text, parent_size, overlap)):
        parent_id = f"{doc_id}::p{parent_index}"
        for child_index, child_text in enumerate(split_text(parent_text, child_size, 0)):
            chunks.append(
                Chunk(
                    id=f"{parent_id}::c{child_index}",
                    doc_id=doc_id,
                    text=child_text,
                    parent_id=parent_id,
                    parent_text=parent_text,
                )
            )
    return chunks
