from openai import OpenAI

from .StructuredOutputAgent import StructuredOutputAgent
from .tools import BaseTool
from .tasks import TaskList


class OrchestratorAgent(StructuredOutputAgent):  # TODO: Finish Implementation
    def check_list(self):
        return self.task_list.tasks

    def edit_list(self, action: str, task_name: str = None):
        pass

    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.0):
        super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, structure=TaskList, structure_name="TaskList", temperature=temperature)

    def call(self, message: str) -> TaskList:
        self.task_list = super().call(message)
        return self.task_list
