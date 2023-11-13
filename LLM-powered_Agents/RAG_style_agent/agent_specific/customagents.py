import os, sys
sys.path.extend(['..', '../..'])
import langchain
from langchain.agents import initialize_agent, AgentType, Tool
from typing import List, Any, Literal
import gradio as gr
import copy


### To get agents
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from utils.agents.get_agents import get_OpenAI_Functions_agent, get_ReAct_agents

### For OpenAI Function agent
# from langchain.tools.render import format_tool_to_openai_function
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain.agents.format_scratchpad import format_to_openai_functions

### Custom tools
from utils.agent_tools.tools.weather_tools import GetFromOpenWeatherMap
from utils.agent_tools.tools.web_search_tools import get_web_search_tools
from utils.agent_tools.tools.math_tools import GetFromWolfram
from utils.agent_tools.tools.time_tools import GetFromDatetimeModule
from utils.agent_tools.tools.python_repl_tools import GetLangChainPythonRepl
from utils.agent_components.configurations import Configurations


### For ReAct agent
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str


def get_tools(config:Configurations, return_tool_dictionary: bool = False)-> List[Tool]:
        search_tool = get_web_search_tools(config)
        weather_tool = GetFromOpenWeatherMap()
        math_tool = GetFromWolfram()
        time_tool = GetFromDatetimeModule()
        python_repl_tool = GetLangChainPythonRepl()

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




def get_custom_agent(llm, config:Configurations)-> langchain.agents.agent.AgentExecutor :

        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        tool_result = get_tools(config)
        if isinstance(tool_result, tuple):
                tools = tool_result[0]
                tool_dictionary = tool_result[1]
        else:
                tools = tool_result

        if config.agent_type.value == 'ReAct':       
                agent = get_ReAct_agents(llm=llm, tools=tools, RAG_style=False)
                system_msg_break_point = 'You have access to the following tools:'


        if config.agent_type.value == 'ReAct_RAG_style':         
                agent = get_ReAct_agents(llm=llm, tools=tools, RAG_style=True)
                system_msg_break_point = 'You have access to the following tools:'


        
        if config.agent_type.value == 'OpenAI_Functions':     
                agent = get_OpenAI_Functions_agent(llm=llm, tools=tools)
                system_msg_break_point = None
        
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        return agent_executor, system_msg_break_point



class Agent:
        def __init__(self, llm, config:Configurations):
                self.llm = llm
                self.config = config
                self.agent_executor, self.system_msg_break_point = get_custom_agent(self.llm, self.config)
                # self.append_sysem_msg("Your final answer should be the same language as the query (It is ok to use English at intermediate steps).")
                self.original_system_msg = (self.system_msg+' ')[:-1]
                self.delete_scratchpad_logs()

        @property
        def system_msg(self)-> str:
                for runnable_comp in self.agent_executor.agent.runnable:
                        if runnable_comp[0]=='middle':
                                return runnable_comp[1][0].messages[0].prompt.template
        

        def __call__(self, query)-> str:
                return (self.agent_executor.invoke({'input':query}, {"metadata": {"agent_type": self.config.agent_type.value}}))['output']


        def get_ruannble_comp(self, target: Literal['prompt', 'llm'])->Any:
                for runnable_comp in self.agent_executor.agent.runnable:
                        if runnable_comp[0]=='middle':
                                component = runnable_comp[1]
                                break
                if target == 'prompt':
                        prompt = component[0]
                        return prompt 
                elif target == 'llm':
                        try:
                                llm = component[1].bound
                        except AttributeError as e:
                                llm = component[1]
                        return llm

        
        def append_sysem_msg(self, msg: str)-> str:
                try:
                        if self.system_msg_break_point != None:
                                systme_msg_lit = self.system_msg.split(self.system_msg_break_point) 
                                appended_system_msg = systme_msg_lit[0].strip() + ' ' + msg + ' ' + self.system_msg_break_point + ' ' + systme_msg_lit[1].strip()
                        else:
                                appended_system_msg = self.system_msg + ' ' + msg

                        self.get_ruannble_comp('prompt').messages[0].prompt.template = appended_system_msg
                        gr.Info("Appending system message succeeded!")
                        return self.system_msg                      
                except Exception as e:
                        raise gr.Error(e)

        def reset_system_msg(self)-> str:
                try:
                        self.get_ruannble_comp('prompt').messages[0].prompt.template = (self.original_system_msg+' ')[:-1]
                        return self.system_msg
                except Exception as e:
                        raise gr.Error(e)




        def set_max_tokens(self, max_tokens:int)-> None:
                self.get_ruannble_comp('llm').max_tokens = max_tokens


        def set_temperature(self, temperature: int)-> None:
                self.get_ruannble_comp('llm').temperature = temperature        

        def delete_scratchpad_logs(self)-> None:
                try:
                        with open(os.path.join(self.config.scratchpad_log_folder, 'scratch_log.log'), 'w') as f:
                                pass
                except FileNotFoundError as e:
                        pass
