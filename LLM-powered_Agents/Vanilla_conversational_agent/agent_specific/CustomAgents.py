import os
import langchain
from langchain.agents import initialize_agent, AgentType, Tool
from typing import List, Any, Literal
import gradio as gr
import copy

### For OpenAI Function agent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

### Custom tools
from AgentTools.weather_tools import GetFromOpenWeatherMap
from AgentTools.web_search_tools import get_web_search_tools
from AgentTools.math_tools import GetFromWolfram
from AgentTools.time_tools import GetFromDatetimeModule
from AgentTools.python_repl_tools import GetLangChainPythonRepl
from agent_specific.configurations import Configurations


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
                return tools, tool_dictionary
        else:
                return tools




def get_custom_agent(llm, config:Configurations)-> langchain.agents.agent.AgentExecutor :

        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        tools = get_tools(config)

        if config.agent_type.value == 'ReAct':         
                prompt = hub.pull("hwchase17/react-multi-input-json")
                human_msg = copy.deepcopy(prompt.messages[1])
                prompt.messages[1] = MessagesPlaceholder(variable_name="chat_history")
                prompt.messages.append(human_msg)                
                prompt = prompt.partial(
                        tools=render_text_description_and_args(tools),
                        tool_names=", ".join([t.name for t in tools]),
                )
                llm_with_stop = llm.bind(stop=["Observation"])
                agent = (
                        {
                                "input": lambda x: x["input"],
                                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
                                "chat_history": lambda x: x["chat_history"],
                        }
                        | prompt | llm_with_stop | JSONAgentOutputParser()
                )  # RunnablePassthrough.assign?

                system_msg_break_point = 'You have access to the following tools:'
                agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        
        
        if config.agent_type.value == 'OpenAI Functions':                
                functions = [format_tool_to_openai_function(f) for f in tools]
                llm_with_functions = llm.bind(functions=functions)

                prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template("You are a helpful assistant. Respond to the human as helpfully and accurately as possible."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])
                agent = RunnablePassthrough.assign(
                        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
                ) | prompt | llm_with_functions | OpenAIFunctionsAgentOutputParser() # This parser contains either agent's intermediate step or final message                
                
                system_msg_break_point = None
                agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

        return agent_executor, system_msg_break_point



class Agent:
        def __init__(self, llm, config:Configurations):
                self.llm = llm
                self.config = config
                self.agent_executor, self.system_msg_break_point = get_custom_agent(self.llm, self.config)
                self.append_sysem_msg("Your final answer should be the same language as the query (It is ok to use English at intermediate steps).")
                self.original_system_msg = (self.system_msg+' ')[:-1]
                self.delete_scratchpad_logs()

        @property
        def system_msg(self)-> str:
                for runnable_comp in self.agent_executor.agent.runnable:
                        if runnable_comp[0]=='middle':
                                return runnable_comp[1][0].messages[0].prompt.template
        

        def __call__(self, query):
                return (self.agent_executor.invoke({'input':query}))['output']


        def get_ruannble_comp(self, target:Literal['prompt', 'bound'])->Any:
                # if self.config.agent_type.value == 'OpenAI Functions':
                for runnable_comp in self.agent_executor.agent.runnable:
                        if runnable_comp[0]=='middle':
                                component = runnable_comp[1]
                                break
                if target == 'prompt':
                        return component[0]
                elif target == 'bound':
                        return component[1].bound

        
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
                        # if self.config.agent_type.value == 'ReAct':
                        #         self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = (self.original_system_msg+' ')[:-1]
                        # if self.config.agent_type.value == 'OpenAI Functions':
                        self.get_ruannble_comp('prompt').messages[0].prompt.template = (self.original_system_msg+' ')[:-1]
                        # gr.Info("Restoring system message succeeded!")
                        return self.system_msg
                except Exception as e:
                        raise gr.Error(e)




        def set_max_tokens(self, max_tokens:int)-> None:
                # if self.config.agent_type.value == 'ReAct':
                #         self.agent_executor.agent.llm_chain.llm.max_tokens = max_tokens
                # if self.config.agent_type.value == 'OpenAI Functions':
                self.get_ruannble_comp('bound').max_tokens = max_tokens


        def set_temperature(self, temperature: int)-> None:
                # if self.config.agent_type.value == 'ReAct':
                #         self.agent_executor.agent.llm_chain.llm.temperature = temperature
                # if self.config.agent_type.value == 'OpenAI Functions':
                self.get_ruannble_comp('bound').temperature = temperature        

        def delete_scratchpad_logs(self)-> None:
                try:
                        with open(os.path.join(self.config.scratchpad_log_folder, 'scratch_log.log'), 'w') as f:
                                pass
                except FileNotFoundError as e:
                        pass
