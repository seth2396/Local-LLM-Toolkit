import pytest
from local_llm_toolkit.agents.BinaryDecisionAgent import BinaryDecisionAgent
from tests.agents.conftest import make_parse_completion

MODEL = "gpt-4o-mini"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_agent(mock_client):
    return BinaryDecisionAgent("Answer 0 or 1.", mock_client, MODEL)

@pytest.fixture
def tf_agent(mock_client):
    return BinaryDecisionAgent("Answer true or false.", mock_client, MODEL, truefalse=True)


# ── Mode selection ─────────────────────────────────────────────────────────────

def test_default_mode_uses_binary_choice_structure(binary_agent):
    assert binary_agent.structure is BinaryDecisionAgent.BinaryChoice

def test_truefalse_mode_uses_tf_choice_structure(tf_agent):
    assert tf_agent.structure is BinaryDecisionAgent.TFChoice


# ── Binary mode (int 0/1) ──────────────────────────────────────────────────────

def test_binary_call_returns_int_one(binary_agent, mock_client):
    parsed = BinaryDecisionAgent.BinaryChoice(value=1)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    assert binary_agent.call("yes or no?") == 1

def test_binary_call_returns_int_zero(binary_agent, mock_client):
    parsed = BinaryDecisionAgent.BinaryChoice(value=0)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    assert binary_agent.call("yes or no?") == 0

def test_binary_call_returns_int_not_object(binary_agent, mock_client):
    parsed = BinaryDecisionAgent.BinaryChoice(value=1)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    result = binary_agent.call("question")
    assert isinstance(result, int)


# ── TrueFalse mode (bool) ──────────────────────────────────────────────────────

def test_tf_call_returns_true(tf_agent, mock_client):
    parsed = BinaryDecisionAgent.TFChoice(value=True)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    assert tf_agent.call("is this true?") is True

def test_tf_call_returns_false(tf_agent, mock_client):
    parsed = BinaryDecisionAgent.TFChoice(value=False)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    assert tf_agent.call("is this true?") is False

def test_tf_call_returns_bool_not_object(tf_agent, mock_client):
    parsed = BinaryDecisionAgent.TFChoice(value=True)
    mock_client.chat.completions.parse.return_value = make_parse_completion(parsed=parsed)
    result = tf_agent.call("question")
    assert isinstance(result, bool)


# ── BinaryChoice validation ────────────────────────────────────────────────────

def test_binary_choice_rejects_value_above_1():
    with pytest.raises(Exception):
        BinaryDecisionAgent.BinaryChoice(value=2)

def test_binary_choice_rejects_negative_value():
    with pytest.raises(Exception):
        BinaryDecisionAgent.BinaryChoice(value=-1)
