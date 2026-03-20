import pytest
from unittest.mock import MagicMock
from openai import OpenAI


# ── Client fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """
    A real OpenAI instance (fake key) with mocked completions methods.
    Passes isinstance(client, OpenAI) without patching builtins.
    """
    client = OpenAI(api_key="fake-key-for-testing")
    client.chat.completions.create = MagicMock()
    client.chat.completions.parse = MagicMock()
    return client


# ── Response builders ──────────────────────────────────────────────────────────

def make_chat_completion(content="response text", finish_reason="stop", tool_calls=None):
    """Build a mock chat completion object."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def make_parse_completion(parsed=None, finish_reason="stop", refusal=None):
    """Build a mock structured-output parse completion object."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.parsed = parsed
    choice.message.refusal = refusal
    choice.message.content = parsed.model_dump_json() if parsed is not None else None
    completion = MagicMock()
    completion.choices = [choice]
    return completion
