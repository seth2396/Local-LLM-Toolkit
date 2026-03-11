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
    assigned: Optional[str] = Field(..., description="Agent assigned to complete the task.")
    dependencies: Optional[list] = Field(..., description="List of dependencies.")
    result: Optional[Any] = Field(default=None, description="The output or result produced after task execution.")

    def mark_complete(self):
        self.status = "completed"

    def mark_in_progress(self):
        self.status = "in-progress"

    def mark_waiting(self):
        self.status = "waiting"


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
    tasks: List[Task] = Field(default_factory=dict, description="A dictionary containing a list of tasks as the key, and description of the task as the value")
