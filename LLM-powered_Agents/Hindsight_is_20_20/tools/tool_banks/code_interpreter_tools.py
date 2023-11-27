from langchain_experimental.tools.python.tool import PythonREPLTool as REPLTool
from langchain.agents import Tool
from pydantic import BaseModel, Field
from typing import Type

def GetPythonREPL()-> Tool:
    tool = PythonREPLTool()
    return tool


class PythonREPL(BaseModel):
    """A Python shell. Use this to execute python commands.\
Input should be a valid python command. If you want to see the output of a value, \
you should print it out with `print(...)`."""
    command : str = Field(description="Valid python command. Must be 'print(...)'") 

class PythonREPLTool:
    # For backward compatibility
    name = 'PythonREPL'
    description="A Python shell. Use this to execute python commands.\
Input should be a valid python command. If you want to see the output of a value, \
you should print it out with `print(...)`."
    args_schema : Type[PythonREPL] = PythonREPL
    
    def __init__(self):
        self.tool = REPLTool()

    def run(self, command):
        return self.tool.run(command)

    async def get_result(self, command):
        return self.run(command)
    
    async def arun(self, command):
        result = await self.get_result(command)
        return result



def get_PythonREPL_schema_and_tool():
    return PythonREPL, PythonREPLTool()



if __name__ == '__main__':
    from langchain.tools.render import format_tool_to_openai_function
    repl_function = format_tool_to_openai_function(GetLangChainPythonRepl())
    print(repl_function)
