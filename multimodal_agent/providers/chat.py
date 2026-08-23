"""Text chat model: Ollama (local), Google Gemini, or any OpenAI-compatible API."""

from __future__ import annotations

from ..config import Settings


def build_chat_model(settings: Settings):
    """Return a tool-calling chat model configured from settings.

    Temperature is pinned to 0 so routing and reflection decisions are
    reproducible across runs. The backend is Ollama by default; "gemini" selects
    Google's hosted API and "openai" any provider speaking the OpenAI chat
    completions API, chosen by ``openai_base_url`` (both are used where a local
    Ollama is not available, e.g. on a Hugging Face Space).
    """
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

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
