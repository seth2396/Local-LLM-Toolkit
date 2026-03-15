from typing import Union
from openai import OpenAI
import json
import logging

from .tools import BaseTool, Tool


def log_call(func):
    """
        Decorator for debugging functions
    """
    def logged_function_call(*args, **kwargs):
        logging.debug(f"\n{func.__name__} called with args={args}, kwargs={kwargs}")
        output = func(*args, **kwargs)
        logging.debug(f"{func.__name__} completed with output = {output}\n")
        return output
    return logged_function_call


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
            raise NotImplementedError(f"Current LLM client is not supported. Supported clients are: {', '.join(BaseAgent.SUPPORTED_CLIENTS)}")
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        if tools:
            if isinstance(tools, BaseTool):
                tools = [tools]
            self.tools = [tool.to_dict() for tool in tools]
            self.tool_dict = {tool.name: tool for tool in tools}
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
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": message}]
        logging.info(f"Calling API with {messages}")
        if not self.stream:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
                stream=self.stream
            )

            tool_response = self._check_and_handle_tool_call(response, messages)
            if tool_response:
                return tool_response.choices[0].message.content

            return response.choices[0].message.content

        else:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
                stream=self.stream
            )

            def generate():
                for chunk in response_stream:
                    if chunk:
                        choice = chunk.choices[0]
                        if choice.delta.tool_calls:
                            response_after_tools = self.__handle_tool_call(responses=choice.delta.tool_calls, messages=messages)
                            for tool_response_chunk in response_after_tools:
                                if tool_response_chunk:
                                    choice = tool_response_chunk.choices[0]
                                    yield choice.delta.content
                            return
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

    def inject(self, message: str, inject: Union[str, list[str]], inject_point_string: str = "{inject}"):
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
            num_injects = len(inject)
        else:
            num_injects = 1

        parts = self.system_prompt.split(inject_point_string)
        if len(parts) - 1 != num_injects:
            raise ValueError("Number of inject placeholders does not match number of injection values.")
        inject_list = inject if isinstance(inject, list) else [inject]
        self.system_prompt = "".join(part + inject_txt for part, inject_txt in zip(parts, inject_list + [""]))
        response = self.call(message)
        self.system_prompt = prompt_holder_var

        return response

    def as_tool(self, name: str, description: str, parameters: dict = None):
        """
            Convert the agent into a reusable tool for integration with other agents.
            This method wraps the agent's call functionality as a Tool object that can be
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
                Tool: A Tool object representing this agent as a tool, containing the agent's
                    call method, name, description, and parameter schema. Can be passed to other
                    agents for invocation.
        """
        if not parameters:
            parameters = {
                "properties": {
                    "message": {"type": "string", "description": "The message content to be sent to the agent."},
                },
                "required": ["message"]
            }

        return Tool(function=self.call, name=name, description=description, parameters=parameters)

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
        if response.choices[0].finish_reason == "tool_calls":
            tool_messages = [messages[-1]]
            tool_calls = 0
            while response.choices[0].finish_reason == "tool_calls" and tool_calls < self.tool_call_limit:
                tools_called = response.choices[0].message.tool_calls
                logging.info(f"Tools called: {tools_called}")
                tool_messages.append({
                    "role": "assistant",
                    "tool_calls": tools_called
                })
                for tool_call in tools_called:
                    tool = self.tool_dict.get(tool_call.function.name)
                    if tool:
                        arguments = json.loads(tool_call.function.arguments)
                        tool_response = tool.call(**arguments)
                        tool_messages.append({"role": "tool", "content": str(tool_response)})
                    else:
                        logging.error(f"{tool_call.function.name} was called but does not exist in the tools supplied to the agent.")
                        tool_messages.append({"role": "tool", "content": "ERROR: invalid request"})
                response = self.client.chat.completions.create(model=self.model, messages=tool_messages, tools=self.tools)
                tool_calls += 1
            return response
        return None

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
        tool_messages = [messages[-1]]
        tool_calls = [{"id": response.id, "type": response.type, "function": {"name": response.function.name, "arguments": response.function.arguments}} for response in responses]
        tool_messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                tool_call = json.loads(tool_call)

            tool = self.tool_dict.get(tool_call['function']['name'])
            if tool:
                arguments = tool_call['function']['arguments']
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                tool_response = tool.call(**arguments)
                tool_messages.append({"role": "tool", "tool_call_id": tool_call['id'], "name": tool_call['function']['name'], "content": str(tool_response)})
            else:
                logging.error(f"{tool_call['function']['name']} was called but does not exist in the tools supplied to the agent.")
                tool_messages.append({"role": "tool", "tool_call_id": tool_call['id'], "name": tool_call['function']['name'], "content": f"InvalidRequest: Tool {tool_call['function']['name']} does not exist."})

            return self.client.chat.completions.create(model=self.model, messages=tool_messages, tools=self.tools, stream=True)
