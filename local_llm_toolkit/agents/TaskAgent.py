from .BaseAgent import BaseAgent
from .tasks import Task


class TaskAgent:
    """
    Executes a single Task using a BaseAgent.

    Builds a prompt from the task's description and information, calls the
    agent, stores the result on the task, and updates the task status.

    Attributes:
        agent (BaseAgent): The agent used to execute the task.
        task (Task): The task to be executed.
    """

    def __init__(self, agent: BaseAgent, task: Task):
        self.agent = agent
        self.task = task

    def execute(self) -> str:
        """
        Execute the task using the agent.

        Marks the task in-progress, sends the task description and context to
        the agent, stores the response in task.result, and marks the task
        complete.

        If the task specifies tools_needed, the executor's available tools are
        temporarily filtered to only those names for the duration of this call,
        then restored afterward.

        Returns:
            str: The agent's response to the task.
        """
        self.task.mark_in_progress()
        prompt = self.task.description
        if self.task.information:
            prompt += f"\n\nContext: {self.task.information}"

        if self.task.tools_needed:
            original_tools = self.agent.tools
            original_tool_dict = getattr(self.agent, "tool_dict", {})
            self.agent.tool_dict = {k: v for k, v in original_tool_dict.items() if k in self.task.tools_needed}
            self.agent.tools = [t for t in original_tools if t.get("function", {}).get("name") in self.task.tools_needed]

        try:
            result = self.agent.call(prompt)
            self.task.result = result
            self.task.mark_complete()
        except Exception as e:
            self.task.mark_failed(reason=str(e))
            result = self.task.result
        finally:
            if self.task.tools_needed:
                self.agent.tools = original_tools
                self.agent.tool_dict = original_tool_dict

        return result
