import base64

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multimodal_agent.agent.runner import (
    _history_to_messages,
    _initial_state,
    encode_image,
)


def test_encode_image_none_returns_none():
    assert encode_image(None) is None


def test_encode_image_bytes_roundtrip():
    encoded = encode_image(b"hello")
    assert base64.b64decode(encoded) == b"hello"


def test_history_to_messages_maps_roles_and_skips_non_text():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": {"path": "img.png"}},  # multimodal turn, skipped
        {"role": "user", "content": ""},
    ]
    messages = _history_to_messages(history)
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]


def test_initial_state_prepends_system_and_appends_question():
    state = _initial_state("what is rrf", None, None)
    assert isinstance(state["messages"][0], SystemMessage)
    assert isinstance(state["messages"][-1], HumanMessage)
    assert state["messages"][-1].content == "what is rrf"
    assert state["reflections"] == 0
