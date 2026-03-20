from pydantic import BaseModel
import logging
from openai import OpenAI

from .BaseAgent import BaseAgent
from .tools import BaseTool


class StructuredOutputAgent(BaseAgent):
    """
        An LLM agent for returning structured outputs based on a given structure

        Attributes:
            system_prompt (str): The system prompt to guide the agents directive
            client (OpenAI): The client for interfacing with the model
            model (str): The model to use for the agent
            output_stucture (BaseModel): A pydantic BaseModel structure for the agent to output
            tools (BaseTool | list[BaseTool]): Provide a single tool or a list of tools which the agent has access to
            structure_name (str): The name of the structure for the output
            temperature (float): The temperature setting for the model, default is 0.0

    """
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, structure: BaseModel = None, structure_name: str = None, temperature: float = 0.0):
        super().__init__(system_prompt, client, model, tools, temperature)
        if not structure:
            raise AttributeError(f"structure attribute is required, but none was provided.")
        elif not issubclass(structure, BaseModel):
            raise AttributeError(f"StructuredOutput agent must have type BaseModel, not {type(structure)}")
        if not structure_name:
            raise AttributeError(f"structure_name attribute is required, but none was provided.")
        elif not isinstance(structure_name, str):
            raise AttributeError(f"StructuredOutput agent must have type str, not {type(structure_name)}")
        self.structure = structure
        self.structure_name = structure_name

    def call(self, message: str):
        """
            Single message call to the llm using the prompt as initialized.
        """
        logging.debug(f"StructuredOutputAgent called with structure: {self.structure_name} and schema: {self.structure.model_json_schema()}")
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": message}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            response_format=self.structure,
            temperature=self.temperature)

        choice = completion.choices[0]

        if choice.finish_reason == "tool_calls":
            raise AssertionError("Tool calls not supported with structured output agent.")

        content = self._extract_content(choice.message)
        return self.structure.model_validate_json(content)
