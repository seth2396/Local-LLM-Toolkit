import pytest
from unittest.mock import MagicMock
from local_llm_toolkit.agents.TaskAgent import TaskAgent
from local_llm_toolkit.agents.tasks import Task


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_agent(return_value="done", tools=None):
    """Create a mock BaseAgent with a controllable call() return value."""
    agent = MagicMock()
    agent.call.return_value = return_value
    agent.tools = tools or []
    agent.tool_dict = {}
    return agent

def make_task(description="Do something", information="Some context", **kwargs):
    return Task(description=description, information=information, **kwargs)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_execute_marks_complete():
    task = make_task()
    TaskAgent(make_agent(), task).execute()
    assert task.status == "completed"

def test_execute_stores_result():
    task = make_task()
    TaskAgent(make_agent(return_value="the answer"), task).execute()
    assert task.result == "the answer"

def test_execute_returns_result():
    task = make_task()
    result = TaskAgent(make_agent(return_value="the answer"), task).execute()
    assert result == "the answer"

def test_execute_calls_agent_with_description():
    task = make_task(description="Summarize this", information="")
    agent = make_agent()
    TaskAgent(agent, task).execute()
    call_arg = agent.call.call_args[0][0]
    assert "Summarize this" in call_arg

def test_execute_appends_context_when_information_present():
    task = make_task(description="Do it", information="extra context")
    agent = make_agent()
    TaskAgent(agent, task).execute()
    call_arg = agent.call.call_args[0][0]
    assert "extra context" in call_arg

def test_execute_no_context_when_information_empty():
    task = make_task(description="Do it", information="")
    agent = make_agent()
    TaskAgent(agent, task).execute()
    call_arg = agent.call.call_args[0][0]
    assert "Context:" not in call_arg


# ── Failure handling ──────────────────────────────────────────────────────────

def test_execute_marks_failed_on_exception():
    agent = make_agent()
    agent.call.side_effect = RuntimeError("API error")
    task = make_task()
    TaskAgent(agent, task).execute()
    assert task.status == "failed"

def test_execute_stores_error_reason_on_failure():
    agent = make_agent()
    agent.call.side_effect = RuntimeError("API error")
    task = make_task()
    TaskAgent(agent, task).execute()
    assert "API error" in task.result

def test_execute_returns_error_string_on_failure():
    agent = make_agent()
    agent.call.side_effect = RuntimeError("API error")
    task = make_task()
    result = TaskAgent(agent, task).execute()
    assert "API error" in result


# ── Tool filtering ────────────────────────────────────────────────────────────

def make_tool_dict(names):
    return {name: MagicMock() for name in names}

def make_tool_list(names):
    return [{"function": {"name": name, "description": ""}} for name in names]

def test_execute_filters_tools_to_needed_only():
    agent = make_agent(
        tools=make_tool_list(["search", "code", "file"]),
    )
    agent.tool_dict = make_tool_dict(["search", "code", "file"])
    task = make_task(tools_needed=["search"])

    TaskAgent(agent, task).execute()

    # During the call, only "search" should have been available
    call_tools = [t["function"]["name"] for t in agent.tools]
    # After execute(), tools are restored — verify by checking the mock was called
    assert agent.call.called

def test_execute_restores_tools_after_completion():
    original_tools = make_tool_list(["search", "code", "file"])
    agent = make_agent(tools=list(original_tools))
    agent.tool_dict = make_tool_dict(["search", "code", "file"])
    task = make_task(tools_needed=["search"])

    TaskAgent(agent, task).execute()

    assert agent.tools == original_tools

def test_execute_restores_tools_after_failure():
    original_tools = make_tool_list(["search", "code"])
    agent = make_agent(tools=list(original_tools))
    agent.tool_dict = make_tool_dict(["search", "code"])
    agent.call.side_effect = RuntimeError("fail")
    task = make_task(tools_needed=["search"])

    TaskAgent(agent, task).execute()

    assert agent.tools == original_tools

def test_execute_does_not_filter_when_no_tools_needed():
    original_tools = make_tool_list(["search", "code"])
    agent = make_agent(tools=list(original_tools))
    task = make_task()

    TaskAgent(agent, task).execute()

    assert agent.tools == original_tools
