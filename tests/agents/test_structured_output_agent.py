import pytest
from pydantic import BaseModel
from local_llm_toolkit.agents.StructuredOutputAgent import StructuredOutputAgent
from tests.agents.conftest import make_parse_completion

MODEL = "gpt-4o-mini"


# ── Test structure ─────────────────────────────────────────────────────────────

class MyOutput(BaseModel):
    answer: str


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(mock_client):
    return StructuredOutputAgent("You return structured output.", mock_client, MODEL, structure=MyOutput, structure_name="MyOutput")


# ── Initialization validation ──────────────────────────────────────────────────

def test_missing_structure_raises(mock_client):
    with pytest.raises(AttributeError):
        StructuredOutputAgent("prompt", mock_client, MODEL, structure_name="name")

def test_non_basemodel_structure_raises(mock_client):
    with pytest.raises(AttributeError):
        StructuredOutputAgent("prompt", mock_client, MODEL, structure=str, structure_name="name")

def test_missing_structure_name_raises(mock_client):
    with pytest.raises(AttributeError):
        StructuredOutputAgent("prompt", mock_client, MODEL, structure=MyOutput)

def test_non_string_structure_name_raises(mock_client):
    with pytest.raises(AttributeError):
        StructuredOutputAgent("prompt", mock_client, MODEL, structure=MyOutput, structure_name=123)

def test_stores_structure(agent):
    assert agent.structure is MyOutput

def test_stores_structure_name(agent):
    assert agent.structure_name == "MyOutput"


# ── call() ────────────────────────────────────────────────────────────────────

def test_call_returns_parsed_object(agent, mock_client):
    parsed = MyOutput(answer="42")
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    result = agent.call("What is the answer?")
    assert result == parsed

def test_call_uses_parse_not_create(agent, mock_client):
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=MyOutput(answer="ok"))
    agent.call("question")
    assert mock_client.chat.completions.parse.called
    assert not mock_client.chat.completions.create.called

def test_call_raises_on_tool_calls_finish_reason(agent, mock_client):
    mock_client.chat.completions.parse.return_value = make_parse_completion(finish_reason="tool_calls")
    with pytest.raises(AssertionError):
        agent.call("question")

def test_call_sends_system_and_user_messages(agent, mock_client):
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=MyOutput(answer="ok"))
    agent.call("my question")
    messages = mock_client.chat.completions.parse.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "my question"


