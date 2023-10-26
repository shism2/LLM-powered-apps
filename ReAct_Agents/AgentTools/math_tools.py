import os
import load_envs
from langchain.utilities import WolframAlphaAPIWrapper
from typing import Type
from langchain.agents import Tool


'''
Wolfram API tool
https://www.wolframalpha.com/
'''
def GetFromWolfram():
    wolf = WolframAlphaAPIWrapper()
    tool = Tool(
        name="Get_from_Wolfram",
        func=wolf.run,
        description="useful for when you need to answer questions about math"
    )
    return tool