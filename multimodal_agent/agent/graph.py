"""LangGraph orchestration: ReAct tool routing plus a reflection loop."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .._content import content_to_text
from ..rag.chunking import Chunk
from .prompts import REFLECTION_PROMPT
from .state import AgentState
from .tools import TOOLS, AnalyzeImage, SearchDocs

REVISION_PREFIX = "Revise your previous answer."

# A citation is a bracketed source slug like [langgraph] or [hybrid-search]. It
# must start with a letter so list indices or code like [0] are never mistaken
# for a source id.
_CITATION_RE = re.compile(r"\[([a-z][a-z0-9._-]+)\]", re.IGNORECASE)


def format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a citable context block for the model."""
    if not chunks:
        return "No relevant documents were found."
    blocks = [f"[{chunk.doc_id}] {chunk.parent_text.strip()}" for chunk in chunks]
    return "\n\n".join(blocks)


def _parse_reflection(content: str) -> tuple[bool, str]:
    """Parse the reviewer's JSON verdict, defaulting to sufficient on any error."""
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return True, ""
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return True, ""
    return bool(data.get("sufficient", True)), str(data.get("critique", ""))


def _last_question(messages: list) -> str:
    """Return the most recent genuine user question, skipping revision prompts."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not message.content.startswith(
            REVISION_PREFIX
        ):
            return message.content
    return ""


def cited_ids(answer: str) -> set[str]:
    """Extract the source ids the answer cites in square brackets."""
    return {match.lower() for match in _CITATION_RE.findall(answer)}


def _gather_grounding(messages: list) -> tuple[str | None, str | None]:
    """Collect the SearchDocs context and image analysis produced this run.

    These are the tool results the agent actually worked from. Feeding them to
    the reviewer is what lets it check faithfulness instead of guessing from the
    answer's wording alone.
    """
    doc_blocks: list[str] = []
    vision_blocks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name == SearchDocs.__name__:
            doc_blocks.append(message.content)
        elif message.name == AnalyzeImage.__name__:
            vision_blocks.append(message.content)
    doc_context = "\n\n".join(doc_blocks) if doc_blocks else None
    vision_desc = "\n\n".join(vision_blocks) if vision_blocks else None
    return doc_context, vision_desc


def deterministic_defects(answer: str, retrieved_ids: list[str]) -> list[str]:
    """Grounding checks that never misjudge a correct answer.

    Pure string checks against the ids SearchDocs actually returned: they can
    only fire on a citation that provably was not retrieved or on a search whose
    result was never cited, so they cannot turn a good answer into a worse one.
    """
    valid = {doc_id.lower() for doc_id in retrieved_ids}
    if not valid:  # no search happened, or it found nothing to cite
        return []
    defects: list[str] = []
    cited = cited_ids(answer)
    if not cited:
        defects.append(
            "You used SearchDocs but cited no source; cite the documents you "
            "relied on by id in square brackets."
        )
    fabricated = sorted(cited - valid)
    if fabricated:
        listed = ", ".join(f"[{doc_id}]" for doc_id in fabricated)
        defects.append(
            f"You cited {listed}, which was not among the retrieved sources; "
            "cite only the retrieved documents and remove the rest."
        )
    return defects


def _image_analysis(vision_desc: str | None, had_image: bool) -> str:
    """Describe, for the reviewer, what the agent knew about the image."""
    if vision_desc:
        return vision_desc
    if had_image:
        return "An image was attached but the assistant did not analyze it."
    return "No image was attached."


def _revision_guidance(retrieved_ids: list[str]) -> str:
    """Re-supply the valid source ids so a revision cannot invent citations."""
    valid = sorted({doc_id.lower() for doc_id in retrieved_ids})
    if not valid:
        return ""
    allowed = ", ".join(f"[{doc_id}]" for doc_id in valid)
    return f" Cite only from these sources: {allowed}."


def build_agent(
    chat_model,
    vision_model,
    retriever,
    *,
    enable_reflection: bool = True,
    max_reflections: int = 1,
):
    """Compile the agent graph with the given model and tool backends injected."""
    model_with_tools = chat_model.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        retrieved_ids: list[str] = []
        for call in last.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            if name == AnalyzeImage.__name__:
                image = state.get("image")
                if not image:
                    content = "No image was provided by the user."
                else:
                    content = vision_model.describe(image, args.get("query", ""))
            elif name == SearchDocs.__name__:
                chunks = retriever.search(args.get("query", ""))
                retrieved_ids.extend(chunk.doc_id for chunk in chunks)
                content = format_chunks(chunks)
            else:
                content = f"Unknown tool: {name}"
            results.append(
                ToolMessage(content=content, name=name, tool_call_id=call["id"])
            )
        return {"messages": results, "retrieved_ids": retrieved_ids}

    def reflect_node(state: AgentState) -> dict:
        answer = content_to_text(state["messages"][-1].content)
        retrieved_ids = state.get("retrieved_ids", [])
        reflections = state.get("reflections", 0) + 1

        # B — deterministic, grounding-based checks first. They cost no model
        # call and can only fire on a provable citation defect, so they never
        # regress a correct answer.
        defects = deterministic_defects(answer, retrieved_ids)

        # A — only pay for the LLM reviewer when the cheap checks pass and we can
        # still act on its verdict. It now sees the retrieved sources and the
        # image analysis, so it judges faithfulness instead of guessing.
        if not defects and reflections <= max_reflections:
            doc_context, vision_desc = _gather_grounding(state["messages"])
            prompt = REFLECTION_PROMPT.format(
                question=_last_question(state["messages"]),
                answer=answer,
                sources=doc_context or "No documentation was retrieved.",
                image_analysis=_image_analysis(vision_desc, bool(state.get("image"))),
            )
            verdict = chat_model.invoke([HumanMessage(content=prompt)])
            sufficient, critique = _parse_reflection(content_to_text(verdict.content))
            # A vague "insufficient" with no concrete fix is treated as sufficient:
            # revising on noise is how a weak reviewer makes an answer worse.
            if not sufficient and critique:
                defects = [critique]

        # Give up once the budget is spent so the draft answer stays final.
        if not defects or reflections > max_reflections:
            return {"reflections": reflections, "needs_revision": False}
        critique_text = " ".join(defects)
        return {
            "reflections": reflections,
            "needs_revision": True,
            "messages": [
                HumanMessage(
                    content=f"{REVISION_PREFIX} {critique_text}"
                    f"{_revision_guidance(retrieved_ids)}"
                )
            ],
        }

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        # Only critique tool-grounded answers. Reflecting on plain conversational
        # replies ("hi") just adds a silent extra model call that keeps the UI
        # locked after the answer is already on screen, with nothing to check.
        used_tools = any(isinstance(m, ToolMessage) for m in state["messages"])
        if enable_reflection and used_tools:
            return "reflect"
        return END

    def route_after_reflect(state: AgentState) -> str:
        return "agent" if state.get("needs_revision") else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")

    agent_routes = {"tools": "tools", END: END}
    if enable_reflection:
        graph.add_node("reflect", reflect_node)
        agent_routes["reflect"] = "reflect"
        graph.add_conditional_edges(
            "reflect", route_after_reflect, {"agent": "agent", END: END}
        )
    graph.add_conditional_edges("agent", route_after_agent, agent_routes)

    return graph.compile()
