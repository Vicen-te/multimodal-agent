"""Build the default agent and run it, synchronously or as a token stream."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import Iterable, Iterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .._content import content_to_text
from ..config import Settings, get_settings
from ..providers.chat import build_chat_model
from ..providers.embeddings import SentenceTransformerEmbedder
from ..providers.vision import build_vision_model
from ..rag.ingest import build_store
from ..rag.retriever import HybridRetriever
from .graph import build_agent
from .prompts import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def _corpus_retriever(embed_model: str, top_k: int, rrf_k: int) -> HybridRetriever:
    """Embed and index the corpus once per process.

    Retrieval does not depend on the chat or vision backend, so agents built
    for different API keys can share the index instead of re-embedding the
    corpus on every build.
    """
    embedder = SentenceTransformerEmbedder(embed_model)
    store = build_store(embedder)
    return HybridRetriever(store, embedder, top_k=top_k, rrf_k=rrf_k)


def build_default_agent(settings: Optional[Settings] = None):
    """Wire real providers and the corpus into a compiled agent graph."""
    settings = settings or get_settings()
    retriever = _corpus_retriever(settings.embed_model, settings.rag_top_k, settings.rrf_k)
    return build_agent(
        build_chat_model(settings),
        build_vision_model(settings),
        retriever,
        enable_reflection=settings.enable_reflection,
        max_reflections=settings.max_reflections,
    )


def encode_image(image) -> Optional[str]:
    """Normalise a PIL image, file path, or raw bytes to a base64 string."""
    if image is None:
        return None
    if isinstance(image, str):
        with open(image, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("ascii")
    if hasattr(image, "save"):  # PIL.Image.Image
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _history_to_messages(history: Optional[Iterable]) -> list:
    messages: list = []
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
        else:
            role, content = turn
        if not isinstance(content, str) or not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


_IMAGE_NOTICE = (
    "\n\n[An image is attached to this message. Call AnalyzeImage to inspect it "
    "before answering anything about its visual content.]"
)


def _initial_state(
    question: str, image_base64: Optional[str], history: Optional[Iterable]
) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_to_messages(history))
    # The text model never sees the image bytes, so it cannot know one was
    # attached unless we tell it. Without this notice the model has to guess from
    # the wording and often skips AnalyzeImage entirely.
    user_content = question + _IMAGE_NOTICE if image_base64 else question
    messages.append(HumanMessage(content=user_content))
    return {"messages": messages, "image": image_base64, "reflections": 0}


def _run_config(settings: Settings) -> dict:
    """Attach a Langfuse callback when configured, disabling it gracefully.

    The handler reads its credentials from the same environment variables the
    settings come from. Any import/version mismatch falls back to no tracing
    rather than breaking the run.
    """
    if not settings.langfuse_enabled:
        return {}
    try:
        from langfuse.langchain import CallbackHandler  # langfuse v3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse v2
        except ImportError:
            return {}
    try:
        return {"callbacks": [CallbackHandler()]}
    except Exception:
        return {}


def run_agent(
    graph,
    question: str,
    image=None,
    history: Optional[Iterable] = None,
    settings: Optional[Settings] = None,
) -> str:
    """Run the agent to completion and return the final answer text."""
    settings = settings or get_settings()
    state = _initial_state(question, encode_image(image), history)
    result = graph.invoke(state, config=_run_config(settings))
    return content_to_text(result["messages"][-1].content)


def run_agent_streaming(
    graph,
    question: str,
    image=None,
    history: Optional[Iterable] = None,
    settings: Optional[Settings] = None,
) -> Iterator[str]:
    """Yield the growing final answer as the agent node streams tokens."""
    settings = settings or get_settings()
    state = _initial_state(question, encode_image(image), history)
    answer = ""
    in_agent_burst = False
    for chunk, metadata in graph.stream(
        state, config=_run_config(settings), stream_mode="messages"
    ):
        node = metadata.get("langgraph_node")
        content = content_to_text(getattr(chunk, "content", ""))
        if node == "agent" and content:
            if not in_agent_burst:
                # A fresh answer burst starts; drop any superseded draft.
                answer = ""
                in_agent_burst = True
            answer += content
            yield answer
        elif node != "agent":
            in_agent_burst = False
    if not answer:
        yield run_agent(graph, question, image, history, settings)
