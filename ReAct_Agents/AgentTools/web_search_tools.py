import os
import load_envs
from langchain.utilities import SerpAPIWrapper
from typing import Type
from langchain.agents import Tool


'''
Google Search API tool
https://serpapi.com/
'''
def GetFromSerpAPI():
    search = SerpAPIWrapper()
    tool = Tool(
        name="Googling_from_SerpAPI",
        func=search.run,
        description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a string object."
    )
    return tool