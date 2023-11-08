from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import Tool

def GetLangChainPythonRepl()-> Tool:
    tool = PythonREPLTool()
    return tool