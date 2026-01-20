from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------- To Do List ----------------------------------------------------------
#TODO: StructuredOutputAgent 
    # Implement tool calls for structured output agent
#TODO: OrchestratorAgent
    # Implement OrchestratorAgent class
    # Might have some pre-implemented tools for task tracking, etc
#TODO: Update Tool handling, currently has a hard limit for number of tool calls
    # Implement dynamic tool call handling
#TODO: Add streaming support
    # Implement streaming response handling
    # Implement streaming tool call handling
    # Implement streaming for ChatAgent
#TODO: Add as tool method to BaseAgent to allow agents to be used as tools in other agents

#--Stretch Goals--
#TODO: Add support for more LLM clients beyond OpenAI
#TODO: Improve error handling and logging throughout
#TODO: Add more detailed docstrings and type hints throughout
#TODO: Add unit tests for all classes and methods
#TODO: Add support for tool usage tracking and analytics
#TODO: Add guardrails for tool calls and agent behavior


# ---------------------------------------------------------- Tool Classes ----------------------------------------------------------
class BaseTool:
    """
        Base Tool Class

        Attributes:
            function (callable): The function to be called when the tool is invoked
            tool_type (str): The type of tool
            name (str): The name of the tool
            description (str): A description of the tool
            parameters (dict): A dictionary defining the parameters for the tool function
    """
    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}):
        self.name = name
        self.function = function
        self.tool_type = tool_type
        self.description = description
        self.parameters = parameters
        self.tool_dict =  {
            "type": tool_type,
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
                }
            }
    
    def call(self, arguments):
        logging.info(f"{self.name}: called with arguments: {arguments}")
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
    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}):
        parameters.update({"type":"object"})
        super().__init__(function = function, tool_type = tool_type, name = name, description = description, parameters = parameters)
    
class Function(Tool):
    """
        Extension of Tool class for function tools
    """
    def __init__(self, function, name: str = "", description: str = "", parameters: dict = {}):
        super().__init__(function = function, tool_type = "function", name=name, description=description, parameters=parameters)
 
# ---------------------------------------------------------- Agent Classes ----------------------------------------------------------
class BaseAgent:
    """
    Docstring for BaseAgent
    
        Attributes:
            system_prompt (str): The system prompt to guide the agents directive 
            client (OpenAI): The client for interfacing with the model
            model (str): The model to use for the agent
            tools (BaseTool | list[BaseTool]): Provide a single tool or a list of tools which the agent has access to
            tool_call_limit (int): The maximum number of tool calls the agent can make in a single call, default is 5
            temperature (float): The temperature setting for the model, default is 0.2
    """
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

    def chat(self, message: str, history: list[dict] = [], stream: bool = False):
        """
            Chat method to handle message calls to the llm using the prompt as initialized.
            Allows for history to be passed in as a list of dicts with role and content keys
        """
        
        messages = [{"role":"system","content":self.system_prompt}] + history + [{"role":"user","content":message}]
        print(messages)
        logging.info(f"Calling API with {messages}")
        if not stream:
            #Standard LLM call with model and messagses, without streaming
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                tools = self.tools, 
                temperature = self.temperature
            )

            #check if there is a tool response. Handle subsequent tool calls
            tool_response = self._check_and_handle_tool_call(response, messages)
            if tool_response:
                return tool_response.choices[0].message.content

            return response.choices[0].message.content
        
        else:

            #Streaming LLM call with model and messages
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                tools = self.tools, 
                temperature = self.temperature, 
                stream=True
                )
            
            #wrap the generator in a function so the upstream functions do not return a generator
            def generate():
                for chunk in response:
                    if chunk: #if none is returned do not yield a chunk
                        choice = chunk.choices[0]
                        if choice.delta.tool_calls:
                            #handle tool calls here or anything that depends on generator yield here.
                            raise NotImplementedError(f"Model attempted to call {choice.tools_calls}. \nTool calls not yet supported for streaming.")
                        yield choice.delta.content
            return generate()
            
    def call(self, message: str, stream: bool = False):
        """
            Single message call to the llm using the prompt as initialized.
        """
        logging.info(f"Agent Called with {message}")
        return self.chat(message=message, history=[], stream=stream)

    def inject(self, message :str, inject: Union[str: list[str]], inject_point_string: str = "{inject}"):
        """
            Injects one or more values into the system prompt before invoking the agent.

            This method temporarily replaces each occurrence of `inject_point_string` in
            the system prompt with the provided injection value(s). If `inject` is a
            single string, exactly one placeholder must exist. If `inject` is a list of
            strings, the number of placeholders must match the number of injection
            values. After constructing the modified prompt, the method calls the agent
            with the given `message` and then restores the original system prompt.

            Parameters
            ----------
            message : str
                The message passed to the agent after prompt injection.
            inject : str or list[str]
                The value(s) to substitute into the system prompt. A single string
                replaces one placeholder; a list replaces multiple placeholders in order.
            inject_point_string : str, optional
                The placeholder token in the system prompt that marks injection points.
                Defaults to "{inject}".

            Returns
            -------
            Any
                The agent's response to the provided message.

            Raises
            ------
            ValueError
                If the number of placeholders in the system prompt does not match the
                number of provided injection values.

            Notes
            -----
            The system prompt is restored to its original state after the agent call,
            ensuring that injections do not persist across invocations.
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

    def as_tool(self, name: str, description: str, parameters: dict):
        """
            Return the agent as a tool for use in other agents.
        """
        raise NotImplementedError("returning agents as tools not yet implemented. Coming soon :)")
        #This needs work
        def tool_function(message: str):
            return self.call(message)
        
        return Tool(function=tool_function, tool_type="agent", name=name, description=description, parameters=parameters)
    
    def _check_and_handle_tool_call(self, response, messages):
        if response.choices[0].finish_reason == "tool_calls": #if no tool is called return None and pass checks
            
            tool_messages = [messages[:-1]] #creata a tool messages list that will just contain the context of the previous message and the tool call request & answers
            tool_messages.append({
                "role": "assistant",
                "tool_calls": response.choices[0].message
            })

            tool_calls = 0
            while response.choices[0].finish_reason == "tool_calls" and tool_calls < self.tool_call_limit:
                #look_up tool in dict and execute tool call
                for tool_call in response.choices[0].message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_response = self.tool_dict.get(tool_call.function.name).call(arguments)
            
                    #add response of the tool call to messages and call model again with added context
                    tool_messages.append({"role":"tool","content": str(tool_response)})
                response = self.client.chat.completions.create(model=self.model, messages=tool_messages, tools = self.tools)
                tool_calls += 1
            logging.debug(response.choices[0].message.content)
            return response
        return None
 
class OrchestratorAgent(BaseAgent):
    def __init__(self):
        raise NotImplementedError("Orchestrator Agent Not yet implemented")
        
class ChatAgent(BaseAgent):
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.2):
        super().__init__(system_prompt, client, model, tools, temperature)
        self.history = []

    def chat(self, message: str, stream = False, history: List[Dict[str,str]] = None):
        #Allows for custom history to be passed which overwrites other otherwise history is stored 
        if history:
            self.history = history
        
        if not stream:
            #Call LLM with history
            response_text = super().chat(message = message, history = self.history, stream = stream)

            #Add model response to history
            self.history += [
                {"role":"user","content":message},
                {"role":"assistant","content":response_text}
            ]
            return response_text
        
        else:
            #Stream reults from chat
            response_gen = super().chat(message = message, history = self.history, stream = stream)
            def generator():
                response_tokens = []
                for token in response_gen:
                    if token:
                        response_tokens.append(token)
                        yield token

                #After streaming completes, update history
                response_text = "".join(response_tokens)
                self.history += [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response_text},
                ]
            return generator()
     
    def inject(self, message, inject):
        response_text = super().inject(message, inject)
        #Add message and model respons to history
        self.history += [{"role":"user","content":message.strip()},{"role":"assistant","content":response_text}]
        return response_text
    
    def reset_history(self):
        logging.info("Chat history reset.")
        self.history = [{"role":"system","content":self.system_prompt}]

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
        logging.debug(f"StructuredOutputAgent called with structure: {self.structure_name} and schema: {self.structure.model_json_schema()}")
        #Created messages using system prompt and user message
        messages = [{"role":"system","content":self.system_prompt},{"role":"user","content":message}]


        completion  = client.chat.completions.parse(
            model=self.model, 
            messages=messages, 
            tools = self.tools, 
            response_format = self.structure, 
            temperature = self.temperature)

        response = completion.choices[0].message

        if completion.choices[0].finish_reason == "tool_calls":
            raise AssertionError("Tool calls not supported with structured output agent.")
            #TODO: implement tool calls for structured output agent
        elif response.refusal:
            raise AssertionError(f"Model refused to provide structured output: {response.refusal.reason}")
        else:
            return response.parsed
 
class BinaryDecisionAgent(StructuredOutputAgent):
    class BinaryChoice(BaseModel):
        value: int = Field(..., ge=0,le=1, description="Binary inidcator for a yes/no or true/false choice. 0 = no/false, 1 = yes/true")

    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.0):
        super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, structure=self.BinaryChoice, structure_name="binary_decision", temperature=temperature)
    
    def call(self, message: str):
        response = super().call(message)
        return response.value

# ---------------------------------------------------------- Example Usage ----------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level="WARNING")

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")


    #---- Setup LLM Client -------
    if openai_api_key :
        print("Using OpenAI API Client")
        openai_base_url = "https://api.openai.com/v1"
        client = OpenAI(api_key=openai_api_key, base_url= openai_base_url)
        model = "gpt-4o-mini"
    else:
        print("Using Ollama Local LLM Client")
        ollama_url = "http://localhost:11434/v1"
        client = OpenAI(api_key="ollama",base_url=ollama_url)
        model = "ministral-3:8b"

    

    #---- Structured Output Agent Test -------
    class TestStruct(BaseModel):
        name: str
        birth_date: str = Field(..., description="month/day/year")
        job: str = Field(..., description="Come up with a fictional job title")
        marital_status: str = Field(..., description="single/married/divorced/widowed")
        favorite_color: str = Field(..., description="Come up with a favorite color")

    StructuredOutputAgentExample = StructuredOutputAgent(
        system_prompt = "You are person generator. Generate some details for a fictional person", 
        client = client, 
        model = model, 
        structure = TestStruct, 
        structure_name = "person_generator", 
        temperature = 0.0)
    print(StructuredOutputAgentExample.call("create me a person!"))

    #---- Binary Decision Agent Test -------
    BinaryDecisionAgentExample = BinaryDecisionAgent(
        system_prompt = "Decide wether the user statement/question is true or false.",
        client = client, 
        model = model,
        temperature = 0)
    #print(BinaryDecisionAgentExample.call("Dogs have legs"))
    #print(BinaryDecisionAgentExample.call("The sky is green"))
    #print(BinaryDecisionAgentExample.call("2+2=5"))

    #---- Chat Agent Test -------
    ChatAgentExample = ChatAgent(
        system_prompt = "You are a helpful assistant that provides concise answers.",
        client = client,
        model = model,
        temperature = 0.2)
    #response = ChatAgentExample.chat("Hello! How are you today?",stream =True)
    #for token in response:
    #   print(token, end="", flush=True)

    # print(ChatAgentExample.history)
    #response = ChatAgentExample.chat("Can you give me a description of a humming bird?",stream =True)
    #for token in response:
    #    print(token, end="", flush=True)