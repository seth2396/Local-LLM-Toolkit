from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI
import json


#TODO: Add implementation of structured output agent
#TODO: Add temperature as a parameter for agents
#TODO: Finish StructuredOutput Agent,
#TODO: Finish BinaryDecision Agent, Maybe it can be an extension of the StructuredOutput with binary decision as structure

class BaseTool:
    """
        Base Tool Class
    """
    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}, logging: bool = False):
        self.name = name
        self.function = function
        self.tool_type = tool_type
        self.description = description
        self.parameters = parameters
        self.logging = logging
        self.tool_dict =  {
            "type": tool_type,
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
                }
            }
    
    def call(self, arguments):
        if self.logging:
            print(f"{self.name} called with arguments: {arguments}")
        return self.function(**arguments)

    def __str__(self):
        return str(self.tool_dict)
    
    def __repr__(self):
        return str(self.tool_dict)

    def to_dict(self):
        return self.tool_dict

class Tool(BaseTool):
    """
        Extension of base tool class for tool types which always should return an object for function arguments
    """
    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}, logging: bool = False):
        parameters.update({"type":"object"})
        super().__init__(function = function, tool_type = tool_type, name = name, description = description, parameters = parameters, logging=logging)
    
class Function(Tool):
    """
        Extension of Tool class for function tools
    """
    def __init__(self, function, name: str = "", description: str = "", parameters: dict = {}, logging: bool = False):
        super().__init__(function = function, tool_type = "function", name=name, description=description, parameters=parameters, logging=logging)
 
# ---------------------------------------------------------- Agent Classes ----------------------------------------------------------
class BaseAgent:
    SUPPORTED_CLIENTS = ["OpenAI"]
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, tool_call_limit: int = 5, temperature: float = 0.2):
        if not isinstance(client, OpenAI):
            raise NotImplementedError(f"Current LLM client is not supported. Supported clients are:{"".join(client,"/n")}")
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        if tools:
            if isinstance(tools,BaseTool):
                tools = [tools]
            self.tools = [tool.to_dict() for tool in tools] #converts tools into a list of dicts expected by "OpenAI"
            self.tool_dict = {tool.name: tool for tool in tools} #creates a dict of the tools for lookup and call {"name":tool} -> tool.call(**args)
        else: 
            self.tools = []

        self.tool_call_limit = tool_call_limit
        self.temperature = temperature

    def call(self, message: str ):
        """
            Single message call to the llm using the prompt as initialized.
        """
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":message}]
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools = self.tools, temperature = self.temperature)

        #check if there is a tool response. Handle subsequent tool calls
        tool_response = self._check_and_handle_tool_call(response, messages)
        if tool_response:
            return tool_response.choices[0].message.content

        return response.choices[0].message.content

    def inject(self, message :str, inject: Union[str: list[str]], inject_point_string: str = "{inject}"):
        """
            Replace each {inject} in the system prompt with provided inject value(s) str or list[str] before calling the agent.
        """
        prompt_holder_var = self.system_prompt
        if isinstance(inject, list):
            num_injects =len(inject)
        else:
            num_injects = 1

        #split prompt by injection parts
        parts = self.system_prompt.split(inject_point_string)
        if len(parts) - 1 != num_injects:
            raise ValueError("Number of inject placeholders does not match number of injection values.")
        # Interleave parts and injects
        self.system_prompt = "".join(part + inject_txt for part, inject_txt in zip(parts, inject + [""]))
        response =  self.call(message)
        self.system_prompt = prompt_holder_var

        return response

    def chat():
        raise NotImplementedError("Load method not implemented.")
    
    def _check_and_handle_tool_call(self, response, messages):
        if response.choices[0].finish_reason == "tool_calls":
            tool_calls = 0
            while response.choices[0].finish_reason == "tool_calls" and tool_calls < self.tool_call_limit:
                #look_up tool in dict and execute to0l call
                for tool_call in response.choices[0].message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_response = self.tool_dict.get(tool_call.function.name).call(arguments)
            
                    #add response to messages and call model again with added context
                    messages.append({"role":"tool","content": str(tool_response)})
                response = self.client.chat.completions.create(model=self.model, messages=messages, tools = self.tools)
                tool_calls += 1
            print(response.choices[0].message.content)
            return response
        return None
 
class OrchestratorAgent(BaseAgent):
    def __init__(self):
        raise NotImplementedError("Load method not implemented.")
        
class ChatAgent(BaseAgent):
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.2):
        super().__init__(system_prompt, client, model, tools, temperature)
        self.history = [{"role":"system","content":self.system_prompt}]

    def chat(self, message):
        #Add message to history before passing context to LLM
        message = {"role":"user","content":message.strip()}
        self.history += [message]


        #LLM call with model and history
        response = self.client.chat.completions.create(messages=self.history, model=self.model, tools=self.tools, temperature = self.temperature)
        tool_response = self._check_and_handle_tool_call(response, [message])
        if tool_response:
            response_text = tool_response.choices[0].message.content
        else: 
            response_text = response.choices[0].message.content

        #Add model response to history
        self.history += [{"role":"assistant","content":response_text}]
        return response_text
    
    def inject(self, message, inject):
        response_text = super().inject(message, inject)
        #Add message and model respons to history
        self.history += [{"role":"user","content":message.strip()},{"role":"assistant","content":response_text}]
        return response_text

class StructuredOutputAgent(BaseAgent):
    """
        An LLM agent for returning structured outputs based on a given structure

        Attributes:
            system_prompt (str): The system prompt to guide the agents directive
            client (OpenAI): The client for interfacing with the model
            model (str): The model to use for the agent
            output_stucture (BaseModel): A pydantic BaseModel structure for the agent to output
            tools (BaseTool | list[BaseTool]): Provide a single tool or a list of tools which the agent has access to
    """
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, structure: BaseModel = None, structure_name: str = None, temperature: float = 0.2):
        super().__init__(system_prompt, client, model, tools, temperature)
        if not structure:
            raise AttributeError(f"structure attribute is required, but none was provided.")
        elif  not issubclass(structure, BaseModel):
            raise AttributeError(f"StructuredOutput agent must have type BaseModel, not {type(structure)}")
        if not structure_name: 
            raise AttributeError(f"structure_name attribute is required, but none was provided.")
        elif  not isinstance(structure_name, str):
            raise AttributeError(f"StructuredOutput agent must have type str, not {type(structure_name)}")
        self.structure = structure
        self.structure_name = structure_name

    def call(self, message: str ):
        """
            Single message call to the llm using the prompt as initialized.
        """
        #Created messages using system prompt and user message
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":message}]

        #Create response format
        response_format = {
            "type":"json_schema",
            "json_schema" : {
                "name":self.structure_name,
                "schema":self.structure.model_json_schema()
            },
            "strict": True
        }

        response = self.client.chat.completions.create(model=self.model, messages=messages, tools = self.tools, response_format = response_format, temperature = self.temperature)

        #check if there is a tool response. Handle subsequent tool calls
        tool_response = self._check_and_handle_tool_call(response, messages)
        if tool_response:
            return tool_response.choices[0].message.content

        return response.choices[0].message.content

class BinaryDecisionAgent(StructuredOutputAgent):
    class TestObj(BaseModel):
        binary_indicator: int = Field(..., ge=0,le=1, description="Binary inidcator for a yes/no or true/false decision" )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Decision",
            "schema": {
                "type": "object",
                "properties": {"value": {"type": "integer", "enum": [0, 1]}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
        "strict": True}

    def call(self, message):
        message = "Here is the user message: \n" + message + "respond only with json content."
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "enum": [0, 1]},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["value"],
            "additionalProperties": False,
            }
        
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":message}]
        response = self.client.chat.completions.create(model=self.model, messages=messages, response_format=schema,  temperature = self.temperature)
        """
        try:
            return int(response.choices[0].message.content)
        except:
            raise AssertionError(f"Could not convert response to integer. /n {response.choices[0].message.content}")
        """
        return response.choices[0].message.content

if __name__ == "__main__":

    ollama_url = "http://localhost:11434/v1"
    client = OpenAI(api_key="ollama",base_url=ollama_url)
    model = "ministral-3:3b"
    system_prompt = "You are person generator. Generate some details for a fictional person"

    class TestStruct(BaseModel):
        name: str
        birth_date: int
        job: str
        marital_status: str

    agent = StructuredOutputAgent(system_prompt = system_prompt, client = client, model = model, structure = TestStruct, structure_name = "person_generator", temperature = 0)
    print(agent.call("create me a person!"))