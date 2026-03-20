import logging
from openai import OpenAI

from .BaseAgent import BaseAgent
from .StructuredOutputAgent import StructuredOutputAgent
from .TaskAgent import TaskAgent
from .tools import BaseTool
from .tasks import TaskList, FeasibilityCheck


class OrchestratorAgent(StructuredOutputAgent):
    """
    Plans a task list from a high-level goal and dispatches each task to an executor agent.

    Uses structured output to generate a TaskList from the goal, then executes
    each task in order using a TaskAgent, respecting declared dependencies.

    Attributes:
        system_prompt (str): System prompt guiding task decomposition.
        client (OpenAI): The LLM client.
        model (str): Model to use for planning.
        tools (BaseTool | list[BaseTool]): Optional tools available during planning.
        temperature (float): Sampling temperature. Defaults to 0.0 for deterministic plans.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a planning agent. Your job is to decompose a high-level goal into an ordered list of discrete, actionable tasks.\n\n"
        "For each task:\n"
        "- description: A single, clear action statement.\n"
        "- information: Any context, constraints, or prior results the executor will need.\n"
        "- dependencies: List the descriptions of any tasks that must complete before this one. Leave empty if none.\n"
        "- tools_needed: List the names of tools required to complete this task. Leave empty if none.\n\n"
        "Keep tasks focused and atomic. Do not bundle multiple actions into one task."
    )

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, client: OpenAI = None, model: str = None, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.0):
        super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, structure=TaskList, structure_name="TaskList", temperature=temperature)
        self.task_list: TaskList | None = None

    def call(self, message: str) -> TaskList:
        """
        Generate a TaskList from a high-level goal.

        Args:
            message (str): The goal or objective to decompose into tasks.

        Returns:
            TaskList: The structured list of tasks produced by the planner.
        """
        self.task_list = super().call(message)
        return self.task_list

    def _check_feasibility(self, goal: str, executor: BaseAgent, tools_section: str) -> FeasibilityCheck:
        """
        Assess whether a goal is achievable given the executor's available tools.

        Args:
            goal (str): The goal to evaluate.
            executor (BaseAgent): The agent whose tools define the available capabilities.
            tools_section (str): Pre-formatted string listing available tools.

        Returns:
            FeasibilityCheck: Structured assessment with feasible flag, reason, and missing capabilities.
        """
        prompt = (
            "Assess whether the following goal can be accomplished using only the available tools listed below.\n"
            f"{tools_section}\n\nGoal: {goal}"
        )
        messages = [{"role": "system", "content": "You are a capability assessment agent. Determine if a goal is achievable with the provided tools."}, {"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "FeasibilityCheck",
                    "schema": FeasibilityCheck.model_json_schema(),
                    "strict": True,
                }
            },
            temperature=0.0
        )
        content = self._extract_content(completion.choices[0].message)
        return FeasibilityCheck.model_validate_json(content)

    def run(self, goal: str, executor: BaseAgent) -> TaskList:
        """
        Plan and execute all tasks for a given goal.

        Injects the executor's available tools into the system prompt so the
        planner knows what to assign per task, generates a TaskList, then
        dispatches each task to the executor in order, respecting dependencies.

        Args:
            goal (str): The high-level objective to accomplish.
            executor (BaseAgent): The agent used to execute each task.

        Returns:
            TaskList: The completed task list with results populated on each task.
        """
        tool_lines = [
            f"- {t['function']['name']}: {t['function'].get('description', '')}"
            for t in executor.tools
        ]
        tools_section = ("\n\nAvailable tools:\n" + "\n".join(tool_lines)) if tool_lines else ""

        check = self._check_feasibility(goal, executor, tools_section=tools_section)
        if not check.feasible:
            raise ValueError(
                f"Goal is not feasible: {check.reason}"
                + (f" Missing: {', '.join(check.missing_capabilities)}" if check.missing_capabilities else "")
            )

        original_prompt = self.system_prompt
        if tools_section:
            self.system_prompt += tools_section.replace("Available tools:", "Available tools for task execution (use these names in tools_needed):")

        task_list = self.call(goal)
        self.system_prompt = original_prompt
        completed = set()

        for task in task_list.tasks:
            if task.dependencies:
                unmet = [d for d in task.dependencies if d not in completed]
                if unmet:
                    logging.warning(f"Skipping '{task.description}': unmet dependencies {unmet}")
                    task.mark_waiting()
                    continue

            result = TaskAgent(executor, task).execute()
            completed.add(task.description)
            logging.info(f"Task complete: '{task.description}' → {result[:80] if result else None}")

        return task_list
