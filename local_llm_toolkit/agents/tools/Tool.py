from .BaseTool import BaseTool

class Tool(BaseTool):
    """
        Extension of base tool class for tool types which always should return an object for function arguments
    """
    def __init__(self, function, tool_type: str = "", name: str = "", description: str = "", parameters: dict = {}):
        parameters.update({"type":"object"})
        super().__init__(function = function, tool_type = "function", name = name, description = description, parameters = parameters)