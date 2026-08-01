from multimodal_agent.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.text_model == "qwen3.5:4b"
    assert settings.vision_model == "qwen3-vl:8b"
    assert settings.rrf_k == 60
    assert settings.langfuse_enabled is False


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL", "llama3.2:1b")
    monkeypatch.setenv("RAG_TOP_K", "8")
    monkeypatch.setenv("ENABLE_REFLECTION", "false")
    settings = Settings.from_env()
    assert settings.text_model == "llama3.2:1b"
    assert settings.rag_top_k == 8
    assert settings.enable_reflection is False


def test_langfuse_enabled_requires_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert Settings.from_env().langfuse_enabled is True
