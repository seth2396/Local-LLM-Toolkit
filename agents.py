from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------- To Do List ----------------------------------------------------------

#--Known Bugs--
#TODO: Fix messages format for tool calls to openai without streaming. Need same format as in streaming version

#--Primary Goals--
#TODO: OrchestratorAgent
    # Implement OrchestratorAgent class
    # Finish Implementation
#TODO: TaskAgent
    # Implement TaskAgent class
#TODO: Update Tool handling, currently has a hard limit for number of tool calls
    # Implement dynamic tool call handling
#TODO: Add as tool method to BaseAgent to allow agents to be used as tools in other agents
#TODO: LLM's are sometimes sending  '' as arguments, need to handle for this

#--Stretch Goals--
#TODO: Add support for more LLM clients beyond OpenAI
#TODO: Improve error handling and logging throughout
#TODO: Add more detailed docstrings and type hints throughout
#TODO: Add unit tests for all classes and methods
#TODO: Add support for tool usage tracking and analytics
#TODO: Add guardrails for tool calls and agent behavior
#TODO: Update as_tool function for structured ouputs, so callers can choose what they want to return from the structure


def log_call(func):
    """
        Decorator for debuging functions
    """
    def logged_function_call(*args, **kwargs):
        logging.debug(f"\n{func.__name__} called with args={args}, kwargs={kwargs}")
        output = func(*args, **kwargs)
        logging.debug(f"{func.__name__} completed with output = {output}\n")
        return output
    return logged_function_call

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
    
    def call(self,*args, **kwargs):
        logging.info(f"{self.name}: called with arguments: {kwargs}")
        return self.function(**kwargs)

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
    description: str = Field(...,description="A description of the task that needs to be done.")
    information: str = Field(...,description="Additional information or context supplied to complete the task.")
    status: str = Field(default="pending",description="Completion status of the task.")
    assigned: Optional[str] = Field(...,description="Agent assigned to complete the task.")
    dependencies: Optional[list] = Field(...,description="List of dependencies.")
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
    tasks: List[Task] = Field(default_factory=dict,description="A dictionary containing a list of tasks as the key, and description of the task as the value")

# ---------------------------------------------------------- Agent Classes ----------------------------------------------------------
class BaseAgent:
    """
    Docstring for BaseAgent
    
        Attributes:
            system_prompt (str): The system prompt to guide the agents directive. [Required]
            client (OpenAI): The client for interfacing with the model. [Required]
            model (str): The model to use for the agent. [Required]
            tools (BaseTool | list[BaseTool]): Provide a single tool or a list of tools which the agent has access to. [default: None]
            tool_call_limit (int): The maximum number of tool calls the agent can make in a single call. [default: 5]
            temperature (float): The temperature setting for the model. [default: 0.2]
            stream (bool): Toggle whether the agent streams back information or returns all at once. [default: False]

    """
    SUPPORTED_CLIENTS = ["OpenAI"]
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, tool_call_limit: int = 5, temperature: float = 0.2, stream: bool = False):
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
        self.stream = stream

    def chat(self, message: str, history: list[dict] = []):
        """
            Chat method to handle message calls to the LLM using the initialized prompt.
            This method supports both streaming and non-streaming modes. It maintains conversation
            history and can invoke tools/function calls when needed.
            Args:
                message (str): The user message to send to the LLM.
                history (list[dict], optional): Conversation history as a list of dictionaries with 
                    'role' and 'content' keys. Defaults to an empty list.
            Returns:
                str or Generator: 
                    - If streaming is disabled: Returns the LLM response as a string.
                    - If streaming is enabled: Returns a generator that yields response chunks as strings.
                    - If a tool call is made: Handles the tool invocation and returns the tool response
                    or subsequent LLM response.
            Raises:
                Logs API calls with the formatted messages for debugging purposes.

            Examples:
                Non-streaming:
                    response = agent.chat("What is 2+2?")
                    print(response)
                Streaming:
                    response_generator = agent.chat("Explain quantum computing")
                    for chunk in response_generator:
                        print(chunk, end='', flush=True)
        """
        
        messages = [{"role":"system","content":self.system_prompt}] + history + [{"role":"user","content":message}]
        logging.info(f"Calling API with {messages}")
        if not self.stream:
            #Standard LLM call with model and messagses, without streaming
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                tools = self.tools, 
                temperature = self.temperature,
                stream = self.stream
            )

            #check if there is a tool response. Handle subsequent tool calls
            tool_response = self._check_and_handle_tool_call(response, messages)
            if tool_response:
                return tool_response.choices[0].message.content

            return response.choices[0].message.content
        
        else:

            #Streaming LLM call with model and messages
            response_stream = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                tools = self.tools, 
                temperature = self.temperature, 
                stream = self.stream
                )
            #wrap the generator in a function so the upstream functions do not return a generator
            def generate():
                for chunk in response_stream:
                    if chunk: #if none is returned do not yield a chunk
                        choice = chunk.choices[0]
                        if choice.delta.tool_calls: #if a tool call is encountered, handle the tool call and generate the response
                            response_after_tools = self.__handle_tool_call(responses = choice.delta.tool_calls, messages = messages)
                            for tool_response_chunk in response_after_tools:
                                if tool_response_chunk:
                                    choice = tool_response_chunk.choices[0]
                                    yield  choice.delta.content
                            return #execution is done- return
                        else:
                            yield choice.delta.content
            return generate()
            
    def call(self, message: str):
        """
            Execute a single message call to the language model using the initialized prompt.

            Args:
                message (str): The input message to send to the language model.

            Returns:
                The response from the language model for the given message.

            Note:
                This method initializes an empty conversation history for each call,
                meaning each invocation is independent and does not retain context
                from previous messages.
        """
        logging.info(f"Agent Called with {message}")
        return self.chat(message=message, history=[])

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

    def as_tool(self, name: str, description: str, parameters: dict = None):
        """
            Convert the agent into a reusable tool for integration with other agents.
            This method wraps the agent's call functionality as a Function object that can be
            invoked by other agents. It allows the agent to be used as a composable component
            in multi-agent systems.
            Args:
                name (str): The name of the tool. Used to identify the tool when called by other agents.
                description (str): A human-readable description of what the tool does. Used by agents
                    to understand when and how to use this tool.
                parameters (dict, optional): A JSON Schema dictionary defining the tool's input parameters.
                    If not provided, defaults to a single "message" string parameter.
                    Expected structure:
                    {
                            "<param_name>": {"type": "<type>", "description": "<description>"},
                            ...
                        "required": ["<param_name>", ...]
            Returns:
                Function: A Function object representing this agent as a tool, containing the agent's
                    call method, name, description, and parameter schema. Can be passed to other
                    agents for invocation.
        """
        if not parameters:
            parameters = {
                "properties": {
                    "message": {"type":"string", "description":"The message content to be sent to the agent."},
                },
                "required":["message"]
            }

        return Function(function=self.call, name=name, description=description, parameters=parameters)
    
    def _check_and_handle_tool_call(self, response, messages):
        """
            Check if the model response contains tool calls and handle them iteratively.
            
            This method processes tool calls from the model response, executes the requested tools,
            and re-invokes the model with tool results until no more tool calls are needed or the
            tool call limit is reached.
            
            Args:
                response: The model response object containing finish_reason and message data.
                messages (list): The conversation message history, with the last message being the tool call request.
            
            Returns:
                The final model response object after all tool calls have been processed and resolved,
                or None if the initial response does not contain tool calls.
            
            Raises:
                json.JSONDecodeError: If tool arguments cannot be parsed as valid JSON.
            
            Notes:
                - Logs information about called tools and errors for invalid tool references.
                - Respects the tool_call_limit to prevent infinite loops.
                - Maintains conversation context by building a tool_messages list throughout execution.
        """
        if response.choices[0].finish_reason == "tool_calls": #if no tool is called return None and pass checks
            tool_messages = [messages[-1]] #creata a tool messages list that will just contain the context of the previous message and the tool call request & answers
            tool_calls = 0
            while response.choices[0].finish_reason == "tool_calls" and tool_calls < self.tool_call_limit:
                tools_called = response.choices[0].message.tool_calls #get list of tools called
                logging.info(f"Tools called: {tools_called}")
                #Add tool call to tool messages
                tool_messages.append({
                    "role": "assistant",
                    "tool_calls": tools_called
                })
                #look up tool called in dict and execute tool call
                for tool_call in tools_called:
                    tool = self.tool_dict.get(tool_call.function.name)
                    if tool:
                        arguments = json.loads(tool_call.function.arguments) #get arguments supplied by llm
                        tool_response = tool.call(**arguments) #call tool and store it to tool_response
                        tool_messages.append({"role":"tool","content": str(tool_response)}) #add response of the tool call to messages and call model again with added context
                    else:
                        logging.error(f"{tool_call.function.name} was called but does not exist in the tools supplied to the agent.")
                        tool_messages.append({"role":"tool","content": "ERROR: invalid request"})
                response = self.client.chat.completions.create(model=self.model, messages=tool_messages, tools = self.tools)
                tool_calls+=1
            return response
        return None

    def  __handle_tool_call(self,responses, messages):
        def __handle_tool_call(self, responses, messages):
            """
            Handle tool calls from the LLM by executing the requested tools and returning responses.
            This method processes tool calls made by the language model, executes the corresponding
            tools with the provided arguments, and returns the model's response to the tool outputs.
            Args:
                responses: A list of tool call response objects from the LLM, each containing
                          an id, type, and function details (name and arguments).
                messages: The conversation message history, with the last message being the
                         context for the tool call request.
            Returns:
                A streaming chat completion response from the API after processing all tool calls
                and sending the results back to the model.
            Raises:
                json.JSONDecodeError: If tool call or argument strings cannot be parsed as JSON.
            Note:
                - Constructs messages in the format expected by OpenAI's API.
                - Logs an error if a requested tool does not exist in the tool_dict.
                - Returns an error message to the model if a tool cannot be found.
                - Streaming is enabled for the returned response.
            """
        tool_messages = [messages[-1]] #creat a tool messages list that will just contain the context of the previous message and the tool call request & answers
        #construct tool call message from response
        tool_calls = [{"id":response.id,"type":response.type,"function":{"name":response.function.name,"arguments":response.function.arguments}} for response in responses]
        tool_messages.append({"role": "assistant","tool_calls": tool_calls}) #add tool call to history, expected by openAI

        for tool_call in tool_calls: #For each tool called, 
            if isinstance(tool_call, str): #if the tool_call is given as a string
                tool_call = json.loads(tool_call) #Try to unpack the string

            
            tool = self.tool_dict.get(tool_call['function']['name']) #look the tool in the tool_dict by name
            if tool: #if the tool exists
                arguments = tool_call['function']['arguments'] #get arguments supplied by LLM
                if isinstance(arguments, str): #If arguments are returned as a string,
                    arguments = json.loads(arguments)   #Try and unpack the string

                tool_response = tool.call(**arguments) #Call tool using arguments supplied by llm
                #construct tool response message expected by openAI
                tool_messages.append({"role":"tool", "tool_call_id":tool_call['id'], "name":tool_call['function']['name'],"content": str(tool_response)}) #Create message using tool's response
            else: #if the tool does not exist (should not happen)
                logging.error(f"{tool_call.function.name} was called but does not exist in the tools supplied to the agent.") #Log error
                tool_messages.append({"role":"tool", "tool_call_id":tool_call['id'], "name":tool_call['function']['name'],"content": f"InvalidRequest: Tool {tool_call['function']['name']} does not exist."}) #return response that the tool does not exist

            return self.client.chat.completions.create(model=self.model, messages=tool_messages, tools = self.tools, stream=True) #return response as streaming object, stream must be true here.
     
class ChatAgent(BaseAgent):
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.2, stream: bool = False):
        super().__init__(system_prompt = system_prompt, client = client, model = model, tools = tools, temperature = temperature, stream = stream)
        self.history = []

    def chat(self, message: str, history: List[Dict[str,str]] = None):
        #Allows for custom history to be passed which overwrites other otherwise history is stored 
        if history:
            self.history = history
        
        if not self.stream:
            #Call LLM with history
            response_text = super().chat(message = message, history = self.history)

            #Add model response to history
            self.history += [
                {"role":"user","content":message},
                {"role":"assistant","content":response_text}
            ]
            return response_text
        
        else:
            #Stream reults from chat
            response_gen = super().chat(message = message, history = self.history)
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
    
    def clear_history(self):
        logging.info("Chat history reset")
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

class OrchestratorAgent(StructuredOutputAgent): #TODO: Finish Implementation
    #This might just be an implementation of a list creator, might rename as necessary. orchestrator shouldnt output lists.
    def check_list(self):
        return self.task_list.tasks

    def edit_list(self, action: str, task_name: Optional[str]):
        pass
    
    #check_list_tool = Function(check_list, )

    #edit_list_tool = Function(edit_list, )

    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.0):
        super().__init__(system_prompt = system_prompt, client = client, model = model, tools = tools, structure = TaskList, structure_name ="TaskList", temperature = temperature)
        

    def call(self, message: str) -> TaskList:
        self.task_list = super().call(message)
        return self.task_list
       
class TaskAgent: #TODO: Finish Implementation
    """
        A wrapper class that takes an agent and a task
    """
    def __init__(self, agent: BaseAgent, task: Task):
        self.agent = agent
        self.task = task
    
    def execute(self, *args, **kwargs):
        self.task.mark_in_progress()
        
# ---------------------------------------------------------- Example Usage ----------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level="ERROR")

    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")


    #---- Setup LLM Client -------
    if openai_api_key:
        print("Using OpenAI API Client")
        openai_base_url = "https://api.openai.com/v1"
        client = OpenAI(api_key=openai_api_key, base_url= openai_base_url)
        model = "gpt-4o-mini"
    else:
        print("Using Ollama Local LLM Client")
        ollama_url = "http://localhost:11434/v1"
        client = OpenAI(api_key="ollama",base_url=ollama_url)
        model = "ministral-3:3b"



    #---- Chat Agent Test -------
    print("----Chat Stream Test----")
    ChatAgentExample = ChatAgent(
        system_prompt = "You are a helpful assistant that provides concise answers.",
        client = client,
        model = model,
        temperature = 0.2,
        stream = True)
    response = ChatAgentExample.chat("Hello! How are you today?")
    #for token in response:
    #   print(token, end="", flush=True)

    # print(ChatAgentExample.history)
    #response = ChatAgentExample.chat("Can you give me a description of a humming bird?",stream =True)
    #for token in response:
    #    print(token, end="", flush=True)

    prompt = "Based on the users query, break up the query into subtasks that need to be done in order to complete the query"

    OrchestratorAgentExample = OrchestratorAgent(
        system_prompt = "Based on the users query, break up the query into subtasks that need to be done in order to complete the query",
        client = client, 
        model = model,
        temperature = 0.0
    )
    #TaskList = OrchestratorAgentExample.call("Can you collect all of the jobs from the feature search database that ave been built in 2024")
    #print(TaskList.model_dump_json(indent = 2))

    BinaryDecisionAgentExample = BinaryDecisionAgent(
        system_prompt = "Decide whether the user statement/question is true or false.",
        client = client, 
        model = model,
        temperature = 0.0,
        truefalse = True)
    #print(f"Dogs have legs:{BinaryDecisionAgentExample.call("Dogs have legs")}")
    #print(BinaryDecisionAgentExample.call("The sky is green"))
    #print(BinaryDecisionAgentExample.call("2+2=5"))

    TrueFalseTool = BinaryDecisionAgentExample.as_tool(name = "TrueFalse", description="Decides if a statement is true or false")
    chatbot = ChatAgent(
        system_prompt = "You must decide if the user is telling the truth or a lie. use your tool to see if the user is lying, if they are tell them they're a liar. if not say you trust them.",
        client = client,
        model = model,
        temperature = 0.2,
        tools = TrueFalseTool,
        stream = True)

    print("----Agents as tool and tool stream test----")
    response = chatbot.chat("Are polar bears good swimmers?")
    for token in response:
        print(token, end="", flush=True)
    print("\n")

    response = chatbot.chat("What was my last question?")
    for token in response:
        print(token, end="", flush=True)
    print("\n")