import inspect
from .BaseTool import BaseTool


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class Tool(BaseTool):
    """
    Extension of BaseTool for function-type tools.

    Parameters are always wrapped in a JSON Schema object, which is required
    by the OpenAI function-calling API.

    Use Tool.register() as a decorator to create a Tool directly from a typed
    function without writing the schema by hand.
    """

    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}):
        parameters.update({"type": "object"})
        super().__init__(function=function, tool_type="function", name=name, description=description, parameters=parameters)

    @classmethod
    def register(cls, description: str = ""):
        """
        Decorator that converts a typed function into a Tool.

        Inspects the function's type annotations to build the JSON Schema
        parameter definition automatically. Parameters without default values
        are marked as required.

        Args:
            description (str): Description of what the tool does. Shown to the
                LLM to help it decide when to call this tool.

        Returns:
            Tool: A Tool instance wrapping the decorated function.

        Example:
            @Tool.register(description="Search the web for information")
            def web_search(query: str, max_results: int = 5) -> str:
                ...
        """
        def decorator(func):
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                json_type = _TYPE_MAP.get(param.annotation, "string")
                properties[param_name] = {"type": json_type, "description": ""}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            parameters = {"properties": properties, "required": required}
            return cls(function=func, name=func.__name__, description=description, parameters=parameters)

        return decorator