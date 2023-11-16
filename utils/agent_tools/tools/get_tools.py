from utils.agent_tools.tools.tool_bank import *
from utils.agent_components.configurations import Configurations
from typing import List, Optional
from langchain.agents import Tool

def get_tool_list(config: Optional[Configurations]=None, return_tool_dictionary: bool = False)-> List[Tool]:
        if config == None:
                config = Configurations()
        search_tool = web_search_tools.get_web_search_tools(config)
        weather_tool = weather_tools.GetFromOpenWeatherMap()
        math_tool = math_tools.GetFromWolfram()
        time_tool = time_tools.GetFromDatetimeModule()
        python_repl_tool = python_repl_tools.GetLangChainPythonRepl()

        # tools = [
        #         search_tool, weather_tool, math_tool, time_tool, python_repl_tool
        # ]
        tools = [
                search_tool, weather_tool, time_tool, python_repl_tool
        ]

        if return_tool_dictionary:
                tool_dictionary  = { tool.name:tool for tool in tools}
                return (tools, tool_dictionary)
        else:
                return tools