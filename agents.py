from typing import Any, List, Dict, Optional, Union
from openai import OpenAI
import json


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
 

#Agent Classes
class BaseAgent:
    SUPPORTED_CLIENTS = ["OpenAI"]
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, tool_call_limit: int = 5):
        if not isinstance(client, OpenAI):
            raise NotImplementedError(f"Current LLM client is not supported. Supported clients are:{"".join(client,"/n")}")
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        if isinstance(tools,BaseTool):
            tools = [tools]
        self.tools = [tool.to_dict() for tool in tools] #converts tools into a list of dicts expected by "OpenAI"
        self.tool_dict = {tool.name: tool for tool in tools} #creates a dict of the tools for lookup and call {"name":tool} -> tool.call(**args)
        self.tool_call_limit = tool_call_limit

    def call(self, message: str ):
        """
            Single message call to the llm using the prompt as initialized.
        """
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":message}]
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools = self.tools)

        #check if there is a tool response. Handle subsequent tool calls
        tool_response = self._check_and_handle_tool_call(response, messages)
        if tool_response:
            return tool_response.choices[0].message.content

        """
        print(response.choices[0])
        #Handle tool calling
        tool_calls = 0
        while response.choices[0].finish_reason == "tool_calls" and tool_calls < self.tool_call_limit:
            #look_up tool in dict and execute to0l call
            for tool_call in response.choices[0].message.tool_calls:
                    print(f"tool_called: {tool_call.function.name}")
                    arguments = json.loads(tool_call.function.arguments)
                    tool_response = self.tool_dict.get(tool_call.function.name).call(arguments)
                    print(tool_response)

            #add response to messages and call model again with added context
            messages.append({"role":"tool","content": str(tool_response)})
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools = self.tools)
            tool_calls += 1
        """

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
            return response
        return None
 
class OrchestratorAgent(BaseAgent):
    def __init__(self):
        raise NotImplementedError("Load method not implemented.")

class BinaryDecisionAgent(BaseAgent):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "only_value",
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
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, response_format=schema)
        """
        try:
            return int(response.choices[0].message.content)
        except:
            raise AssertionError(f"Could not convert response to integer. /n {response.choices[0].message.content}")
        """
        return response.choices[0].message.content
        

class ChatAgent(BaseAgent):
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None):
        super().__init__(system_prompt, client, model, tools)
        self.history = [{"role":"system","content":self.system_prompt}]

    def chat(self, message):
        #Add message to history before passing context to LLM
        message = {"role":"user","content":message.strip()}
        self.history += [message]


        #LLM call with model and history
        response = self.client.chat.completions.create(messages=self.history, model=self.model, tools=self.tools)
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


def main():
 pass

if __name__ == "__main__":
    main()