"""Gradio frontend for the multimodal agent (entry point for Hugging Face Spaces)."""

from __future__ import annotations

from dataclasses import replace

import gradio as gr

from multimodal_agent.agent.runner import build_default_agent, run_agent_streaming
from multimodal_agent.config import get_settings

# Backends a visitor can bring a key for: Gemini, or any provider that speaks the
# OpenAI chat completions API, picked by its base URL.
_PROVIDERS = [
    ("Gemini", "gemini"),
    ("OpenAI-compatible (OpenAI, Groq, OpenRouter, Mistral, DeepSeek, xAI, Anthropic...)", "openai"),
]
_BASE_URLS = [
    ("OpenAI", "https://api.openai.com/v1"),
    ("Groq", "https://api.groq.com/openai/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
    ("Mistral", "https://api.mistral.ai/v1"),
    ("DeepSeek", "https://api.deepseek.com/v1"),
    ("xAI", "https://api.x.ai/v1"),
    ("Anthropic", "https://api.anthropic.com/v1"),
]
# Suggested model ids; any other id can be typed in too.
_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
_DEFAULT_MODEL = get_settings().gemini_text_model
_MODEL_CHOICES = [
    *([] if _DEFAULT_MODEL in _GEMINI_MODELS else [_DEFAULT_MODEL]),
    *_GEMINI_MODELS,
    *_OPENAI_MODELS,
]

_agent = None

# Appended to every reply so it is always clear which backend, model, and key
# answered; stripped from the history before it goes back to the model.
_FOOTER = "\n\n_backend: "


def _backend_label(settings, own_key: bool) -> str:
    if settings.llm_provider == "gemini":
        return f"gemini / {settings.gemini_text_model} / {'your key' if own_key else 'shared key'}"
    if settings.llm_provider == "openai":
        return f"openai-compatible / {settings.openai_model} / {'your key' if own_key else 'server key'}"
    return f"ollama / {settings.text_model} / local"


def _strip_footers(history):
    cleaned = []
    for turn in history or []:
        content = turn.get("content") if isinstance(turn, dict) else None
        if isinstance(content, str) and _FOOTER in content:
            turn = {**turn, "content": content.split(_FOOTER, 1)[0]}
        cleaned.append(turn)
    return cleaned


def get_agent(provider: str = "gemini", api_key: str = "", base_url: str = "", model: str = ""):
    """Return (agent, settings, own_key): the shared agent, or one bound to the visitor's key.

    A visitor-supplied key only wires the models for that request and is never
    stored or logged. Without a key, only a Gemini model switch applies, on the
    shared key. The corpus index is shared either way, so the per-request build
    costs a graph compile, not a re-embedding.
    """
    global _agent
    provider = (provider or "gemini").strip().lower()
    key = (api_key or "").strip()
    base_url = (base_url or "").strip()
    model = (model or "").strip()
    overrides = {}
    if key and provider == "openai":
        overrides.update(llm_provider="openai", openai_api_key=key)
        if base_url:
            overrides["openai_base_url"] = base_url
        if model and model not in _GEMINI_MODELS:  # an untouched Gemini pick means "default"
            overrides["openai_model"] = model
    elif key:
        overrides.update(llm_provider="gemini", google_api_key=key)
        if model:
            overrides.update(gemini_text_model=model, gemini_vision_model=model)
    elif provider == "gemini" and model and model != _DEFAULT_MODEL:
        overrides.update(gemini_text_model=model, gemini_vision_model=model)
    settings = replace(get_settings(), **overrides) if overrides else get_settings()
    if overrides:
        return build_default_agent(settings), settings, bool(key)
    if _agent is None:
        _agent = build_default_agent()
    return _agent, settings, False


def chat_fn(message, history, provider="gemini", api_key="", base_url="", model=""):
    """Stream the agent's answer for a multimodal chat turn."""
    text = message.get("text", "") if isinstance(message, dict) else str(message)
    files = message.get("files", []) if isinstance(message, dict) else []
    image = files[0] if files else None

    if not text and not image:
        yield "Please type a question, attach an image, or both."
        return

    agent, settings, own_key = get_agent(provider, api_key, base_url, model)
    label = _backend_label(settings, own_key)
    print(f"[backend] {label}", flush=True)
    try:
        answer = ""
        for partial in run_agent_streaming(
            agent, text, image=image, history=_strip_footers(history)
        ):
            answer = partial
            yield partial
        yield f"{answer}{_FOOTER}{label}_"
    except Exception as exc:
        # With visitor keys, a bad key or an exhausted quota is an expected
        # outcome; answer with the reason, scrubbed of the key, instead of a
        # stack trace.
        detail = str(exc)
        key = (api_key or "").strip()
        if key:
            detail = detail.replace(key, "***")
        yield f"The model call failed ({type(exc).__name__}): {detail[:400]}"


demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=True,
    # `fill_height` lets the app own the viewport height so only the chatbot
    # scrolls; `elem_id` scopes the CSS below that caps message-image height (a
    # tall image was the trigger for a second, nested scrollbar).
    chatbot=gr.Chatbot(height=560, elem_id="chatbot"),
    fill_height=True,
    editable=True,  # users can edit a past message to regenerate from that point
    title="Multimodal Agent",
    description=(
        "Upload an image, ask a question, or both. The agent decides whether to "
        "analyze the image, search the documentation, or both, and reflects on its "
        "answer before sending it. The hosted demo shares a small free quota; if "
        "it runs dry, bring your own key (Gemini or any OpenAI-compatible "
        'provider) or switch model under "Use your own API key or model".'
    ),
    additional_inputs=[
        gr.Dropdown(
            label="Provider",
            choices=_PROVIDERS,
            value="gemini",
            info="Which API your key belongs to.",
        ),
        gr.Textbox(
            label="API key (optional)",
            type="password",
            placeholder="Used only for your requests, never stored",
        ),
        gr.Dropdown(
            label="Base URL (OpenAI-compatible only)",
            choices=_BASE_URLS,
            value=None,
            allow_custom_value=True,
            info="Pick a provider or paste any OpenAI-compatible endpoint; empty means OpenAI.",
        ),
        gr.Dropdown(
            label="Model",
            choices=_MODEL_CHOICES,
            value=_DEFAULT_MODEL,
            allow_custom_value=True,
            info=(
                "Used for text and vision, so pick a vision-capable model. Gemini "
                "models each have their own free quota, so switching one also works "
                "around a dry quota on the shared key."
            ),
        ),
    ],
    additional_inputs_accordion="Use your own API key or model",
    # With additional inputs, each example is [message, *additional values].
    # Leaving those values as None keeps the examples as clickable chips inside
    # the chat (a concrete value would move them to a table below it), and a
    # click then uses whatever key and model the fields currently hold.
    examples=[
        [{"text": "How does LangGraph work?", "files": []}, None, None, None, None],
        [{"text": "What is reciprocal rank fusion?", "files": []}, None, None, None, None],
    ],
)


# The chatbot root (#chatbot) carries an inline `overflow: auto` on top of its
# fixed 560px height, so it scrolls in addition to the inner `.bubble-wrap`
# message list -> two scrollbars. Force the root to hide overflow (needs
# `!important` to beat the inline style) so only the message list scrolls, and cap
# message-image height so a tall photo does not blow up the bubble. Gradio 6 takes
# app CSS via launch().
_CSS = """
#chatbot { overflow: hidden !important; }
#chatbot img { max-height: 320px; width: auto; }
"""

# Launched unguarded, matching the Spaces Gradio template: the app starts whether
# the host runs this file as a script or imports it.
demo.launch(css=_CSS)
