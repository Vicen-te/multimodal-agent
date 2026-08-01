from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multimodal_agent.agent.graph import (
    _parse_reflection,
    build_agent,
    cited_ids,
    deterministic_defects,
)
from multimodal_agent.agent.prompts import SYSTEM_PROMPT
from multimodal_agent.rag.chunking import Chunk

from ._fakes import FakeChat, StubRetriever, StubVision


def _state(question, image=None):
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        "image": image,
        "reflections": 0,
    }


def _tool_call(name, query, call_id="call1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def _retriever():
    # A single chunk with doc_id "doc", so "[doc]" is the one valid citation and
    # anything else is fabricated. (Real corpus ids are multi-char slugs too.)
    return StubRetriever([Chunk("doc::p0::c0", "doc", "child", "doc::p0", "parent")])


def _graph(responses, retriever=None, max_reflections=1):
    return build_agent(
        FakeChat(responses),
        StubVision(),
        retriever or _retriever(),
        enable_reflection=True,
        max_reflections=max_reflections,
    )


# --- verdict parsing -------------------------------------------------------


def test_parse_reflection_valid_json():
    assert _parse_reflection('{"sufficient": false, "critique": "fix it"}') == (False, "fix it")


def test_parse_reflection_defaults_to_sufficient_on_garbage():
    assert _parse_reflection("not json at all") == (True, "")


# --- deterministic citation guard (unit) -----------------------------------


def test_cited_ids_extracts_slugs_and_ignores_indices():
    assert cited_ids("see [langgraph] and [hybrid-search], not [0] or [i]") == {
        "langgraph",
        "hybrid-search",
    }


def test_deterministic_defects_flag_fabricated_citation():
    assert deterministic_defects("uses [ghost]", ["doc"])


def test_deterministic_defects_flag_uncited_search():
    assert deterministic_defects("no citation here", ["doc"])


def test_deterministic_defects_pass_for_valid_citation():
    assert deterministic_defects("grounded in [doc]", ["doc"]) == []


def test_deterministic_defects_skip_when_no_search():
    # No ids retrieved (vision-only or no-tool answer): nothing to validate.
    assert deterministic_defects("a bracketed [thing]", []) == []


# --- grounded review path (A) ----------------------------------------------


def test_grounded_review_accepts_a_cited_answer():
    # Clean answer -> reviewer is consulted -> it accepts -> run ends.
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="answer grounded in [doc]"),
        AIMessage(content='{"sufficient": true, "critique": ""}'),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "answer grounded in [doc]"
    assert result["reflections"] == 1


def test_grounded_review_can_request_one_revision():
    # No deterministic defect, so the reviewer runs and asks for a concrete fix.
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="answer one [doc]"),
        AIMessage(content='{"sufficient": false, "critique": "mention the fusion constant"}'),
        AIMessage(content="answer two [doc]"),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "answer two [doc]"
    assert result["reflections"] == 2


def test_vague_insufficient_verdict_does_not_revise():
    # "insufficient" with an empty critique is treated as sufficient: revising on
    # noise is exactly how a weak reviewer degrades a correct answer.
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="answer [doc]"),
        AIMessage(content='{"sufficient": false, "critique": ""}'),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "answer [doc]"
    assert result["reflections"] == 1


def test_garbage_verdict_keeps_the_draft():
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="answer [doc]"),
        AIMessage(content="the reviewer rambled without json"),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "answer [doc]"
    assert result["reflections"] == 1


# --- deterministic guard drives the loop without the reviewer (B) -----------


def test_fabricated_citation_forces_revision_without_the_reviewer():
    # No JSON verdict is scripted: the loop must revise on the deterministic
    # defect alone (a fabricated [ghost]), then stop once corrected.
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="see [ghost]"),
        AIMessage(content="corrected, see [doc]"),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "corrected, see [doc]"
    assert result["reflections"] == 2


def test_uncited_search_forces_revision_without_the_reviewer():
    graph = _graph([
        _tool_call("SearchDocs", "q"),
        AIMessage(content="an answer with no citation"),
        AIMessage(content="now cited [doc]"),
    ])
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "now cited [doc]"
    assert result["reflections"] == 2


# --- budget and routing ----------------------------------------------------


def test_reflection_cap_stops_the_loop():
    # The reviewer never accepts; the cap must end the run after the budget.
    graph = _graph(
        [
            _tool_call("SearchDocs", "q"),
            AIMessage(content="answer 1 [doc]"),
            AIMessage(content='{"sufficient": false, "critique": "more"}'),
            AIMessage(content="answer 2 [doc]"),
        ],
        max_reflections=1,
    )
    result = graph.invoke(_state("q"))
    assert result["messages"][-1].content == "answer 2 [doc]"
    assert result["reflections"] == 2


def test_direct_answer_without_tools_skips_reflection():
    # A plain conversational reply uses no tools, so reflection must be skipped:
    # only one model response is scripted, and the run must not ask for a second.
    graph = _graph([AIMessage(content="hello, how can I help?")])
    result = graph.invoke(_state("hi"))
    assert result["messages"][-1].content == "hello, how can I help?"
    assert result["reflections"] == 0
