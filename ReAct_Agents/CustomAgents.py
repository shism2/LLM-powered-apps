import os
import langchain
from langchain.agents import initialize_agent, AgentType
from langchain.schema import FunctionMessage

### Custom tools
from AgentTools.weather_tools import GetFromOpenWeatherMap
from AgentTools.web_search_tools import GetFromSerpAPI
from AgentTools.math_tools import GetFromWolfram
from AgentTools.time_tools import GetFromDatetimeModule

tools = [
            GetFromSerpAPI(), GetFromWolfram(), GetFromOpenWeatherMap(), GetFromDatetimeModule()
        ]


def OpenAIFunctionCallAgent(llm, verbose=True)-> langchain.agents.agent.AgentExecutor :
        return initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=verbose)



def ZeroShotReActAgent(llm, verbose=True)-> langchain.agents.agent.AgentExecutor :
        return initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)