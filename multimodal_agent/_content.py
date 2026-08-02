"""Normalize chat-message content to plain text across model backends."""

from __future__ import annotations


def content_to_text(content) -> str:
    """Flatten a message's ``content`` to a plain string.

    Ollama returns content as a string, but Gemini and other newer backends can
    return a list of content blocks (dicts with a ``text`` field). Joining them
    here lets the rest of the pipeline assume plain text regardless of provider.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)
