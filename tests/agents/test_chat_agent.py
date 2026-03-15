import pytest
from unittest.mock import MagicMock
from local_llm_toolkit.agents.ChatAgent import ChatAgent
from tests.agents.conftest import make_chat_completion

MODEL = "gpt-4o-mini"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(mock_client):
    return ChatAgent("You are helpful.", mock_client, MODEL)


# ── Initialization ─────────────────────────────────────────────────────────────

def test_history_starts_empty(agent):
    assert agent.history == []


# ── chat() history management ─────────────────────────────────────────────────

def test_chat_appends_user_and_assistant_to_history(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("hello back")
    agent.chat("hello")
    assert agent.history[-2] == {"role": "user", "content": "hello"}
    assert agent.history[-1] == {"role": "assistant", "content": "hello back"}

def test_chat_accumulates_history_across_turns(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("turn 1")
    agent.chat("message 1")
    mock_client.chat.completions.create.return_value = make_chat_completion("turn 2")
    agent.chat("message 2")
    assert len(agent.history) == 4

def test_chat_history_sent_to_api_on_second_turn(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("first")
    agent.chat("first message")
    mock_client.chat.completions.create.return_value = make_chat_completion("second")
    agent.chat("second message")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles

def test_chat_with_explicit_history_replaces_history(agent, mock_client):
    injected = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "yes"}]
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.chat("new message", history=list(injected))  # pass a copy — chat() mutates history in-place
    assert agent.history[:2] == injected

def test_chat_returns_response_string(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("the response")
    result = agent.chat("question")
    assert result == "the response"


# ── clear_history() ───────────────────────────────────────────────────────────

def test_clear_history_resets_to_system_prompt(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.chat("hello")
    agent.clear_history()
    assert agent.history == [{"role": "system", "content": agent.system_prompt}]

def test_clear_history_on_fresh_agent(agent):
    agent.clear_history()
    assert agent.history == [{"role": "system", "content": agent.system_prompt}]


# ── inject() ─────────────────────────────────────────────────────────────────

def test_inject_appends_to_history(mock_client):
    agent = ChatAgent("You are a {inject} assistant.", mock_client, MODEL)
    mock_client.chat.completions.create.return_value = make_chat_completion("injected response")
    agent.inject("hello", "helpful")
    assert any(m["role"] == "assistant" and m["content"] == "injected response" for m in agent.history)

def test_inject_user_message_in_history(mock_client):
    agent = ChatAgent("You are a {inject} assistant.", mock_client, MODEL)
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.inject("hello", "helpful")
    assert any(m["role"] == "user" for m in agent.history)


# ── Streaming ─────────────────────────────────────────────────────────────────

def test_streaming_returns_generator(mock_client):
    agent = ChatAgent("prompt", mock_client, MODEL, stream=True)

    chunk1, chunk2 = MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "hello "
    chunk1.choices[0].delta.tool_calls = None
    chunk2.choices[0].delta.content = "world"
    chunk2.choices[0].delta.tool_calls = None
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    result = agent.chat("hi")
    import types
    assert isinstance(result, types.GeneratorType)

def test_streaming_yields_tokens(mock_client):
    agent = ChatAgent("prompt", mock_client, MODEL, stream=True)

    chunk1, chunk2 = MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "hello "
    chunk1.choices[0].delta.tool_calls = None
    chunk2.choices[0].delta.content = "world"
    chunk2.choices[0].delta.tool_calls = None
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    tokens = list(agent.chat("hi"))
    assert "hello " in tokens
    assert "world" in tokens

def test_streaming_updates_history_after_consuming(mock_client):
    agent = ChatAgent("prompt", mock_client, MODEL, stream=True)

    chunk1, chunk2 = MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "hello "
    chunk1.choices[0].delta.tool_calls = None
    chunk2.choices[0].delta.content = "world"
    chunk2.choices[0].delta.tool_calls = None
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    list(agent.chat("hi"))  # consume the generator
    assert any(m["content"] == "hello world" for m in agent.history)
