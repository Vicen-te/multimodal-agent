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
        "answer before sending it."
    ),
    examples=[
        {"text": "How does LangGraph work?", "files": []},
        {"text": "What is reciprocal rank fusion?", "files": []},
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
