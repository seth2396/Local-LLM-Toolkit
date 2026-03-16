import pytest
from local_llm_toolkit.agents.tasks import Task, TaskList, FeasibilityCheck


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_task():
    return Task(description="Do something", information="Some context")


# ── Task defaults ─────────────────────────────────────────────────────────────

def test_task_default_status(basic_task):
    assert basic_task.status == "pending"

def test_task_default_result_is_none(basic_task):
    assert basic_task.result is None

def test_task_default_assigned_is_none(basic_task):
    assert basic_task.assigned is None

def test_task_default_dependencies_is_none(basic_task):
    assert basic_task.dependencies is None

def test_task_default_tools_needed_is_none(basic_task):
    assert basic_task.tools_needed is None


# ── Task status transitions ───────────────────────────────────────────────────

def test_mark_in_progress(basic_task):
    basic_task.mark_in_progress()
    assert basic_task.status == "in-progress"

def test_mark_complete(basic_task):
    basic_task.mark_complete()
    assert basic_task.status == "completed"

def test_mark_waiting(basic_task):
    basic_task.mark_waiting()
    assert basic_task.status == "waiting"

def test_mark_failed_sets_status(basic_task):
    basic_task.mark_failed()
    assert basic_task.status == "failed"

def test_mark_failed_with_reason_stores_error(basic_task):
    basic_task.mark_failed(reason="something went wrong")
    assert basic_task.status == "failed"
    assert "something went wrong" in basic_task.result

def test_mark_failed_without_reason_leaves_result_none(basic_task):
    basic_task.mark_failed()
    assert basic_task.result is None

def test_status_transitions_are_independent(basic_task):
    basic_task.mark_in_progress()
    basic_task.mark_complete()
    assert basic_task.status == "completed"


# ── Task optional fields ──────────────────────────────────────────────────────

def test_task_with_dependencies():
    task = Task(description="Step 2", information="", dependencies=["Step 1"])
    assert task.dependencies == ["Step 1"]

def test_task_with_tools_needed():
    task = Task(description="Search", information="", tools_needed=["web_search"])
    assert task.tools_needed == ["web_search"]

def test_task_with_assigned():
    task = Task(description="Do it", information="", assigned="agent-1")
    assert task.assigned == "agent-1"


# ── TaskList ──────────────────────────────────────────────────────────────────

def test_task_list_defaults_to_empty():
    tl = TaskList()
    assert tl.tasks == []

def test_task_list_holds_tasks():
    t1 = Task(description="A", information="")
    t2 = Task(description="B", information="")
    tl = TaskList(tasks=[t1, t2])
    assert len(tl.tasks) == 2
    assert tl.tasks[0].description == "A"

def test_task_list_tasks_are_task_instances():
    tl = TaskList(tasks=[Task(description="X", information="")])
    assert isinstance(tl.tasks[0], Task)


# ── FeasibilityCheck ──────────────────────────────────────────────────────────

def test_feasibility_check_feasible():
    fc = FeasibilityCheck(feasible=True, reason="All tools present")
    assert fc.feasible is True
    assert fc.missing_capabilities == []

def test_feasibility_check_not_feasible():
    fc = FeasibilityCheck(feasible=False, reason="Missing tool", missing_capabilities=["web_search"])
    assert fc.feasible is False
    assert "web_search" in fc.missing_capabilities

def test_feasibility_check_missing_capabilities_defaults_to_empty():
    fc = FeasibilityCheck(feasible=True, reason="OK")
    assert fc.missing_capabilities == []
