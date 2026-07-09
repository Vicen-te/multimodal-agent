"""Text chat model backed by Ollama via langchain-ollama."""

from __future__ import annotations

from ..config import Settings


def build_chat_model(settings: Settings):
    """Return a tool-calling ``ChatOllama`` configured from settings.

    Temperature is pinned to 0 so routing and reflection decisions are
    reproducible across runs.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.text_model,
        base_url=settings.ollama_host,
        temperature=0,
    )
