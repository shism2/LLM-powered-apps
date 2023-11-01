import os
import load_envs
from langchain.utilities import WolframAlphaAPIWrapper
from langchain.agents import Tool


'''
Wolfram API tool
https://www.wolframalpha.com/
'''
def GetFromWolfram()-> Tool:
    wolf = WolframAlphaAPIWrapper()
    tool = Tool(
        name="Get_from_Wolfram",
        func=wolf.run,
        description="Useful for when you need to answer questions about math. Choose this tool over other tools to solve math problems"
        # description="Useful for when you need to answer questions about math"
    )
    return tool