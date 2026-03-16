from typing import Any, List, Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
        A Pydantic BaseModel representing a task that can be assigned to an agent.
        Attributes:
            description (str): A description of the task that needs to be done.
            information (str): Additional information or context supplied to complete the task.
            status (str): Completion status of the task. Defaults to "pending".
            assigned (Optional[str]): Agent assigned to complete the task.
            dependencies (Optional[list]): List of dependencies.
            result (Optional[Any]): The output or result produced after task execution. Defaults to None.
        Methods:
            mark_complete(): Updates the task status to "completed".
            mark_in_progress(): Updates the task status to "in-progress".
            mark_waiting(): Updates the task status to "waiting".
    """
    description: str = Field(..., description="A description of the task that needs to be done.")
    information: str = Field(..., description="Additional information or context supplied to complete the task.")
    status: str = Field(default="pending", description="Completion status of the task.")
    assigned: Optional[str] = Field(default=None, description="Agent assigned to complete the task.")
    dependencies: Optional[list] = Field(default=None, description="List of task descriptions this task depends on.")
    tools_needed: Optional[list[str]] = Field(default=None, description="Names of tools the executor should have access to for this task.")
    result: Optional[Any] = Field(default=None, description="The output or result produced after task execution.")

    def mark_complete(self):
        self.status = "completed"

    def mark_in_progress(self):
        self.status = "in-progress"

    def mark_waiting(self):
        self.status = "waiting"

    def mark_failed(self, reason: str = None):
        self.status = "failed"
        if reason:
            self.result = f"ERROR: {reason}"


class FeasibilityCheck(BaseModel):
    """
    Structured output for a pre-flight feasibility assessment.

    Attributes:
        feasible (bool): Whether the goal can be accomplished with the available tools.
        reason (str): Explanation of why the goal is or is not feasible.
        missing_capabilities (List[str]): Tools or capabilities required but not available.
    """
    feasible: bool = Field(..., description="Whether the goal can be accomplished with the available tools and capabilities.")
    reason: str = Field(..., description="Explanation of the feasibility assessment.")
    missing_capabilities: List[str] = Field(default_factory=list, description="Tools or capabilities that would be needed but are not available.")


class TaskList(BaseModel):
    """
        A Pydantic BaseModel that represents a collection of tasks.

        Attributes:
            tasks (List[Task]): A list of Task objects to be populated by an agent.
                Each task contains information about a specific action or work item
                to be performed.

        Example:
            task_list = TaskList(tasks=[task1, task2, task3])
    """
    tasks: List[Task] = Field(default_factory=list, description="Ordered list of tasks to be executed.")
