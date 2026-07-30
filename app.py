"""Gradio frontend for the multimodal agent (entry point for Hugging Face Spaces)."""

from __future__ import annotations

import gradio as gr

from multimodal_agent.agent.runner import build_default_agent, run_agent_streaming

_agent = None


def get_agent():
    """Build the agent once and reuse it across requests."""
    global _agent
    if _agent is None:
        _agent = build_default_agent()
    return _agent


def chat_fn(message, history):
    """Stream the agent's answer for a multimodal chat turn."""
    text = message.get("text", "") if isinstance(message, dict) else str(message)
    files = message.get("files", []) if isinstance(message, dict) else []
    image = files[0] if files else None

    if not text and not image:
        yield "Please type a question, attach an image, or both."
        return

    for partial in run_agent_streaming(get_agent(), text, image=image, history=history):
        yield partial


demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=True,
    chatbot=gr.Chatbot(height=560, resizable=True),
    editable=True,  # users can edit a past message to regenerate from that point
    title="Multimodal Agent",
    description=(
        "Upload an image, ask a question, or both. The agent decides whether to "
        "analyze the image, search the documentation, or both, and reflects on its "
        "answer before sending it."
    ),
    examples=[
        {"text": "How does LangGraph work?", "files": []},
        {"text": "What is reciprocal rank fusion?", "files": []},
    ],
)


if __name__ == "__main__":
    demo.launch()
