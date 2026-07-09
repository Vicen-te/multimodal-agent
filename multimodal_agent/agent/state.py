"""Shared state passed between graph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State threaded through the agent graph.

    ``messages`` accumulates via the ``add_messages`` reducer and
    ``retrieved_ids`` accumulates via list concatenation; the other keys are
    overwritten on each update. ``image`` holds the user's picture as a base64
    string so the vision tool can read it without it ever entering the text LLM's
    context. ``retrieved_ids`` records the source ids actually returned by
    SearchDocs so the reflection node can tell a real citation from an invented
    one.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    image: Optional[str]
    reflections: int
    needs_revision: bool
    retrieved_ids: Annotated[list[str], operator.add]
