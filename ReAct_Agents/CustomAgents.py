import os
import langchain
from langchain.agents import initialize_agent, AgentType, Tool
# from langchain.callbacks import FileCallbackHandler
# from langchain.load.dump import dumps
from typing import List

### Custom tools
from AgentTools.weather_tools import GetFromOpenWeatherMap
from AgentTools.web_search_tools import get_web_search_tools
from AgentTools.math_tools import GetFromWolfram
from AgentTools.time_tools import GetFromDatetimeModule
from AgentTools.python_repl_tools import GetLangChainPythonRepl
from configurations import Configurations

def get_tools(config:Configurations)-> List[Tool]:
        search_tool = get_web_search_tools(config.search_tool.value)
        weather_tool = GetFromOpenWeatherMap()
        math_tool = GetFromWolfram()
        time_tool = GetFromDatetimeModule()
        python_repl_tool = GetLangChainPythonRepl()

        tools = [
                search_tool, weather_tool, math_tool, time_tool, python_repl_tool
        ]

        return tools



def get_custom_agent(llm, config:Configurations)-> langchain.agents.agent.AgentExecutor :

        if config.agent_type.value == 'zeroshot react':         
                agent_executor = initialize_agent(get_tools(config), llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=config.verbose.value)
                system_message_breakdown = (agent_executor.agent.llm_chain.prompt.messages[0].prompt.template).split(' Format is Action')
                agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = system_message_breakdown[0] + " Your final answer should be the same language as the query.  Format is Action" + system_message_breakdown[1]
        if config.agent_type.value == 'openai functioncall':
                agent_executor = initialize_agent(get_tools(config), llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=config.verbose.value)
        
        return agent_executor



class ReActAgent:
        def __init__(self, llm, config:Configurations):
                self.llm = llm
                self.config = config
                self.agent_executor = get_custom_agent(self.llm, self.config)
                self.system_msg = self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template
                self.original_system_msg = (self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template+' ')[:-1]                
                self.delete_scratchpad_logs()

        def __call__(self, query):
                return self.agent_executor.run(query)

        
        def append_sysem_msg(self, msg: str)-> str:
                system_msg_breakdown = self.system_msg.split(' Format is Action')
                self.system_msg = system_msg_breakdown[0] + f" {msg} Format is Action" + system_msg_breakdown[1]
                self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = self.system_msg
                return self.system_msg

        def reset_system_msg(self)-> str:
                self.system_msg = (self.original_system_msg+' ')[:-1]
                self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = self.system_msg
                return self.system_msg


        def set_max_tokens(self, max_tokens):
                self.agent_executor.agent.llm_chain.llm.max_tokens = max_tokens


        def set_temperature(self, temperature):
                self.agent_executor.agent.llm_chain.llm.temperature = temperature


        def delete_scratchpad_logs(self):
                try:
                        with open(os.path.join(self.config.scratchpad_log_folder, 'scratch_log.log'), 'w') as f:
                                pass
                except FileNotFoundError as e:
                        pass
