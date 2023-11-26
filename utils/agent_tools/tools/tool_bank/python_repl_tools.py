from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import Tool



def GetLangChainPythonRepl()-> Tool:
    tool = PythonREPLTool()
    return tool



if __name__ == '__main__':
    from langchain.tools.render import format_tool_to_openai_function
    repl_function = format_tool_to_openai_function(GetLangChainPythonRepl())
    print(repl_function)
