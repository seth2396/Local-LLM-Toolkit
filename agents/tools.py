import logging


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
        self.tool_dict = {
            "type": tool_type,
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }

    def call(self, *args, **kwargs):
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
        parameters.update({"type": "object"})
        super().__init__(function=function, tool_type="function", name=name, description=description, parameters=parameters)
