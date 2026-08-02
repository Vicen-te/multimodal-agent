"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# override=True so the project's .env wins over stray machine-wide variables
# (e.g. a system OLLAMA_HOST=0.0.0.0 set for `ollama serve`, which clients
# cannot connect to).
load_dotenv(override=True)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot for a process."""

    # Backend for the chat and vision models: "ollama" (local) or "gemini"
    # (hosted Google API, used where a local Ollama is unavailable, e.g. a Space).
    llm_provider: str = "ollama"

    ollama_host: str = "http://localhost:11434"
    text_model: str = "qwen3.5:4b"
    vision_model: str = "qwen3-vl:8b"

    # Google Gemini (used when llm_provider == "gemini"); key from aistudio.google.com.
    google_api_key: str = ""
    gemini_text_model: str = "gemini-3.6-flash"
    gemini_vision_model: str = "gemini-3.6-flash"

    embed_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 4
    rrf_k: int = 60
    enable_reflection: bool = True
    max_reflections: int = 1
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            text_model=os.getenv("TEXT_MODEL", cls.text_model),
            vision_model=os.getenv("VISION_MODEL", cls.vision_model),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            gemini_text_model=os.getenv("GEMINI_TEXT_MODEL", cls.gemini_text_model),
            gemini_vision_model=os.getenv("GEMINI_VISION_MODEL", cls.gemini_vision_model),
            embed_model=os.getenv("EMBED_MODEL", cls.embed_model),
            rag_top_k=_get_int("RAG_TOP_K", cls.rag_top_k),
            rrf_k=_get_int("RRF_K", cls.rrf_k),
            enable_reflection=_get_bool("ENABLE_REFLECTION", cls.enable_reflection),
            max_reflections=_get_int("MAX_REFLECTIONS", cls.max_reflections),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", cls.langfuse_host),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
