import pytest
from unittest.mock import MagicMock
from local_llm_toolkit.agents.OrchestratorAgent import OrchestratorAgent
from local_llm_toolkit.agents.tasks import Task, TaskList, FeasibilityCheck
from tests.agents.conftest import make_parse_completion

MODEL = "gpt-4o-mini"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_feasibility(feasible=True, reason="OK", missing=None):
    return FeasibilityCheck(feasible=feasible, reason=reason, missing_capabilities=missing or [])

def make_task_list(*descriptions):
    tasks = [Task(description=d, information="") for d in descriptions]
    return TaskList(tasks=tasks)

def make_executor(return_value="done", tools=None):
    executor = MagicMock()
    executor.call.return_value = return_value
    executor.tools = tools or []
    executor.tool_dict = {}
    return executor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(mock_client):
    return OrchestratorAgent(client=mock_client, model=MODEL)


# ── Default prompt ─────────────────────────────────────────────────────────────

def test_has_default_system_prompt():
    assert OrchestratorAgent.DEFAULT_SYSTEM_PROMPT != ""

def test_uses_default_prompt_when_none_given(mock_client):
    agent = OrchestratorAgent(client=mock_client, model=MODEL)
    assert agent.system_prompt == OrchestratorAgent.DEFAULT_SYSTEM_PROMPT

def test_accepts_custom_prompt(mock_client):
    agent = OrchestratorAgent(system_prompt="Custom prompt.", client=mock_client, model=MODEL)
    assert agent.system_prompt == "Custom prompt."


# ── call() ────────────────────────────────────────────────────────────────────

def test_call_stores_task_list(agent, mock_client):
    tl = make_task_list("Task A")
    mock_client.chat.completions.create.return_value = make_parse_completion(parsed=tl)
    agent.call("do something")
    assert isinstance(agent.task_list, TaskList)
    assert agent.task_list == tl

def test_call_returns_task_list(agent, mock_client):
    tl = make_task_list("Task A")
    mock_client.chat.completions.create.return_value = make_parse_completion(parsed=tl)
    result = agent.call("do something")
    assert isinstance(result, TaskList)
    assert result == tl


# ── run() feasibility check ───────────────────────────────────────────────────

def test_run_raises_when_not_feasible(agent, mock_client):
    check = make_feasibility(feasible=False, reason="No file tool", missing=["file_tool"])
    mock_client.chat.completions.create.return_value = make_parse_completion(parsed=check)
    with pytest.raises(ValueError, match="not feasible"):
        agent.run("write a file", make_executor())

def test_run_raises_includes_missing_capabilities(agent, mock_client):
    check = make_feasibility(feasible=False, reason="Missing", missing=["web_search"])
    mock_client.chat.completions.create.return_value = make_parse_completion(parsed=check)
    with pytest.raises(ValueError, match="web_search"):
        agent.run("search the web", make_executor())


# ── run() prompt injection ────────────────────────────────────────────────────

def test_run_restores_system_prompt_after_planning(agent, mock_client):
    original_prompt = agent.system_prompt
    executor = make_executor(tools=[{"function": {"name": "search", "description": "Search"}}])
    tl = make_task_list("Task A")

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility(feasible=True)),
        make_parse_completion(parsed=tl),
    ]

    agent.run("do something", executor)
    assert agent.system_prompt == original_prompt

def test_run_injects_tool_names_into_planning_prompt(agent, mock_client):
    executor = make_executor(tools=[{"function": {"name": "search", "description": "Search the web"}}])
    tl = make_task_list("Task A")

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility(feasible=True)),
        make_parse_completion(parsed=tl),
    ]

    agent.run("goal", executor)
    planning_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    system_content = planning_call_messages[0]["content"]
    assert "search" in system_content

def test_run_with_no_tools_does_not_modify_prompt(agent, mock_client):
    original_prompt = agent.system_prompt
    tl = make_task_list("Task A")

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility(feasible=True)),
        make_parse_completion(parsed=tl),
    ]

    agent.run("goal", make_executor(tools=[]))
    planning_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    assert planning_call_messages[0]["content"] == original_prompt


# ── run() task execution ──────────────────────────────────────────────────────

def test_run_executes_all_tasks(agent, mock_client):
    executor = make_executor()
    tl = make_task_list("Task A", "Task B")

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility()),
        make_parse_completion(parsed=tl),
    ]

    agent.run("goal", executor)
    assert executor.call.call_count == 2

def test_run_returns_task_list(agent, mock_client):
    tl = make_task_list("Task A")
    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility()),
        make_parse_completion(parsed=tl),
    ]
    result = agent.run("goal", make_executor())
    assert isinstance(result, TaskList)

def test_run_tasks_are_marked_complete(agent, mock_client):
    tl = make_task_list("Task A", "Task B")
    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility()),
        make_parse_completion(parsed=tl),
    ]
    result = agent.run("goal", make_executor(return_value="result"))
    for task in result.tasks:
        assert task.status == "completed"


# ── run() dependency handling ─────────────────────────────────────────────────

def test_run_skips_task_with_unmet_dependency(agent, mock_client):
    task_a = Task(description="Task A", information="")
    task_b = Task(description="Task B", information="", dependencies=["Task X"])  # X never runs
    tl = TaskList(tasks=[task_a, task_b])

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility()),
        make_parse_completion(parsed=tl),
    ]

    executor = make_executor()
    result = agent.run("goal", executor)
    assert result.tasks[1].status == "waiting"
    assert executor.call.call_count == 1  # only Task A ran

def test_run_executes_task_when_dependency_is_met(agent, mock_client):
    task_a = Task(description="Task A", information="")
    task_b = Task(description="Task B", information="", dependencies=["Task A"])
    tl = TaskList(tasks=[task_a, task_b])

    mock_client.chat.completions.create.side_effect = [
        make_parse_completion(parsed=make_feasibility()),
        make_parse_completion(parsed=tl),
    ]

    result = agent.run("goal", make_executor())
    assert result.tasks[1].status == "completed"
