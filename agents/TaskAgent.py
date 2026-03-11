from .BaseAgent import BaseAgent
from .tasks import Task


class TaskAgent:  # TODO: Finish Implementation
    """
        A wrapper class that takes an agent and a task
    """
    def __init__(self, agent: BaseAgent, task: Task):
        self.agent = agent
        self.task = task

    def execute(self, *args, **kwargs):
        self.task.mark_in_progress()
