from pydantic import BaseModel, Field
from openai import OpenAI

from .StructuredOutputAgent import StructuredOutputAgent
from .tools import BaseTool


class BinaryDecisionAgent(StructuredOutputAgent):

    class BinaryChoice(BaseModel):
        value: int = Field(..., ge=0, le=1, description="Binary inidcator for a yes/no or true/false choice. 0 = no/false, 1 = yes/true")

    class TFChoice(BaseModel):
        value: bool = Field(..., description="True/False inidcator for a yes/no or true/false question.")

    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.0, truefalse: bool = False):
        if truefalse:
            super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, structure=self.TFChoice, structure_name="TrueFalse_decision", temperature=temperature)
        else:
            super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, structure=self.BinaryChoice, structure_name="binary_decision", temperature=temperature)

    def call(self, message: str):
        response = super().call(message)
        return response.value
