from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from multimodal_agent.agent.graph import build_agent, format_chunks
from multimodal_agent.agent.prompts import SYSTEM_PROMPT
from multimodal_agent.rag.chunking import Chunk

from ._fakes import FakeChat, StubRetriever, StubVision


def _tool_call(name, query, call_id="call1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def _state(question, image=None):
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        "image": image,
        "reflections": 0,
    }


def test_format_chunks_emits_citable_sources():
    chunks = [Chunk("c0", "langgraph", "child", "p0", "parent body")]
    rendered = format_chunks(chunks)
    assert "[langgraph]" in rendered
    assert "parent body" in rendered


def test_format_chunks_empty():
    assert "No relevant documents" in format_chunks([])


def test_search_route_calls_retriever():
    chat = FakeChat([
        _tool_call("SearchDocs", "what is rrf"),
        AIMessage(content="RRF fuses rankings [rrf]."),
        AIMessage(content='{"sufficient": true, "critique": ""}'),
    ])
    # The retrieved doc id matches the answer's citation, so the grounded review
    # runs (rather than the deterministic citation guard firing).
    retriever = StubRetriever([Chunk("rrf::p0::c0", "rrf", "child", "rrf::p0", "parent")])
    vision = StubVision()
    graph = build_agent(chat, vision, retriever, enable_reflection=True, max_reflections=1)

    result = graph.invoke(_state("what is rrf"))

    assert retriever.queries == ["what is rrf"]
    assert vision.calls == []
    assert result["messages"][-1].content == "RRF fuses rankings [rrf]."


def test_vision_route_calls_vision_model():
    chat = FakeChat([
        _tool_call("AnalyzeImage", "what is shown"),
        AIMessage(content="The image shows code."),
        AIMessage(content='{"sufficient": true, "critique": ""}'),
    ])
    retriever = StubRetriever()
    vision = StubVision(reply="a function")
    graph = build_agent(chat, vision, retriever, enable_reflection=True, max_reflections=1)

    result = graph.invoke(_state("what is shown", image="ZmFrZQ=="))

    assert vision.calls == [("ZmFrZQ==", "what is shown")]
    assert retriever.queries == []
    assert result["messages"][-1].content == "The image shows code."


def test_vision_tool_without_image_reports_missing():
    chat = FakeChat([
        _tool_call("AnalyzeImage", "what is shown"),
        AIMessage(content="No image to analyze."),
    ])
    vision = StubVision()
    graph = build_agent(chat, vision, StubRetriever(), enable_reflection=False)

    result = graph.invoke(_state("what is shown", image=None))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == "No image was provided by the user."
    assert vision.calls == []


def test_direct_answer_without_tools():
    chat = FakeChat([AIMessage(content="Hello, I can help with images and docs.")])
    graph = build_agent(chat, StubVision(), StubRetriever(), enable_reflection=False)

    result = graph.invoke(_state("hi"))

    assert result["messages"][-1].content.startswith("Hello")
