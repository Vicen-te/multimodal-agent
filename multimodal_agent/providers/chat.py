"""Text chat model: Ollama (local) or Google Gemini (hosted)."""

from __future__ import annotations

from ..config import Settings


def build_chat_model(settings: Settings):
    """Return a tool-calling chat model configured from settings.

    Temperature is pinned to 0 so routing and reflection decisions are
    reproducible across runs. The backend is Ollama by default, or Google
    Gemini when ``llm_provider`` is "gemini" (e.g. on a Hugging Face Space,
    where a local Ollama is not available).
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_text_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.text_model,
        base_url=settings.ollama_host,
        temperature=0,
    )
