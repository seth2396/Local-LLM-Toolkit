import pytest
from unittest.mock import MagicMock
from openai import OpenAI

from local_llm_toolkit.agents.BaseAgent import BaseAgent
from local_llm_toolkit.agents.tools import Tool
from tests.agents.conftest import make_chat_completion

MODEL = "gpt-4o-mini"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(mock_client):
    return BaseAgent("You are helpful.", mock_client, MODEL)


@pytest.fixture
def tool():
    return Tool(function=lambda x: x, name="my_tool", description="Does a thing", parameters={
        "properties": {"x": {"type": "string"}},
        "required": ["x"]
    })


# ── Initialization ─────────────────────────────────────────────────────────────

def test_stores_system_prompt(mock_client):
    agent = BaseAgent("Be concise.", mock_client, MODEL)
    assert agent.system_prompt == "Be concise."

def test_stores_model(mock_client):
    agent = BaseAgent("prompt", mock_client, MODEL)
    assert agent.model == MODEL

def test_stores_temperature(mock_client):
    agent = BaseAgent("prompt", mock_client, MODEL, temperature=0.9)
    assert agent.temperature == 0.9

def test_no_tools_gives_empty_list(mock_client):
    agent = BaseAgent("prompt", mock_client, MODEL)
    assert agent.tools == []

def test_single_tool_wrapped_in_list(mock_client, tool):
    agent = BaseAgent("prompt", mock_client, MODEL, tools=tool)
    assert len(agent.tools) == 1

def test_tool_list_stored_as_dicts(mock_client, tool):
    agent = BaseAgent("prompt", mock_client, MODEL, tools=[tool])
    assert isinstance(agent.tools[0], dict)

def test_tool_dict_keyed_by_name(mock_client, tool):
    agent = BaseAgent("prompt", mock_client, MODEL, tools=[tool])
    assert "my_tool" in agent.tool_dict

def test_non_openai_client_raises():
    with pytest.raises(NotImplementedError):
        BaseAgent("prompt", MagicMock(), MODEL)


# ── call() ─────────────────────────────────────────────────────────────────────

def test_call_returns_response_content(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("hello")
    assert agent.call("hi") == "hello"

def test_call_sends_system_and_user_messages(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion()
    agent.call("test message")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]

def test_call_uses_empty_history(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion()
    agent.call("test")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert len(messages) == 2  # system + user only


# ── inject() ───────────────────────────────────────────────────────────────────

def test_inject_replaces_placeholder(mock_client):
    agent = BaseAgent("You are a {inject} assistant.", mock_client, MODEL)
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.inject("hello", "helpful")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert "helpful" in messages[0]["content"]
    assert "{inject}" not in messages[0]["content"]

def test_inject_restores_original_prompt(mock_client):
    agent = BaseAgent("You are a {inject} assistant.", mock_client, MODEL)
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.inject("hello", "helpful")
    assert agent.system_prompt == "You are a {inject} assistant."

def test_inject_list_replaces_multiple_placeholders(mock_client):
    agent = BaseAgent("Act as {inject} for {inject}.", mock_client, MODEL)
    mock_client.chat.completions.create.return_value = make_chat_completion("ok")
    agent.inject("hello", ["a coder", "Python tasks"])
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert "a coder" in messages[0]["content"]
    assert "Python tasks" in messages[0]["content"]

def test_inject_wrong_count_raises_value_error(mock_client):
    agent = BaseAgent("You are a {inject} assistant.", mock_client, MODEL)
    with pytest.raises(ValueError):
        agent.inject("hello", ["one", "two"])


# ── as_tool() ──────────────────────────────────────────────────────────────────

def test_as_tool_returns_tool_instance(agent):
    t = agent.as_tool("agent_tool", "Runs the agent")
    assert isinstance(t, Tool)

def test_as_tool_name_and_description(agent):
    t = agent.as_tool("agent_tool", "Runs the agent")
    assert t.name == "agent_tool"
    assert t.description == "Runs the agent"

def test_as_tool_callable_delegates_to_agent(agent, mock_client):
    mock_client.chat.completions.create.return_value = make_chat_completion("tool result")
    t = agent.as_tool("agent_tool", "desc")
    result = t.call(message="do something")
    assert result == "tool result"


# ── Tool call handling ─────────────────────────────────────────────────────────

def test_check_and_handle_tool_call_returns_none_when_no_tool_calls(agent):
    completion = make_chat_completion(finish_reason="stop")
    result = agent._check_and_handle_tool_call(completion, [])
    assert result is None

def test_check_and_handle_tool_call_processes_tool_and_returns_final(mock_client, tool):
    agent = BaseAgent("prompt", mock_client, MODEL, tools=[tool])

    tool_call = MagicMock()
    tool_call.function.name = "my_tool"
    tool_call.function.arguments = '{"x": "hello"}'
    first = make_chat_completion(finish_reason="tool_calls", tool_calls=[tool_call])
    first.choices[0].message.tool_calls = [tool_call]

    second = make_chat_completion(content="final answer", finish_reason="stop")
    mock_client.chat.completions.create.return_value = second

    result = agent._check_and_handle_tool_call(first, [{"role": "user", "content": "do it"}])
    assert result.choices[0].message.content == "final answer"
