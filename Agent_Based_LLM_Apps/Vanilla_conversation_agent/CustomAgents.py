import os
import langchain
from langchain.agents import initialize_agent, AgentType, Tool
from typing import List, Any, Literal
import gradio as gr

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
from configurations import Configurations

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
        
        if config.agent_type.value == 'zeroshot react':         
                agent_executor = initialize_agent(get_tools(config), llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=config.verbose.value, memory=memory)
                agent_executor.memory = memory
                system_message_breakdown = (agent_executor.agent.llm_chain.prompt.messages[0].prompt.template).split(' Format is Action')
                agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = system_message_breakdown[0] + " Your final answer should be the same language as the query (It is ok to use English at intermediate steps).  Format is Action" + system_message_breakdown[1]
        
        
        
        if config.agent_type.value == 'openai functioncall':
                tools = get_tools(config)
                functions = [format_tool_to_openai_function(f) for f in tools]
                llm_with_functions = llm.bind(functions=functions)

                prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template("You are a helpful assistant. Respond to the human as helpfully and accurately as possible. Your final answer should be the same language as the query (It is ok to use English at intermediate steps)."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])
                agent = RunnablePassthrough.assign(
                        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
                ) | prompt | llm_with_functions | OpenAIFunctionsAgentOutputParser() # This parser contains either agent's intermediate step or final message                
                agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

        return agent_executor



class ReActAgent:
        def __init__(self, llm, config:Configurations):
                self.llm = llm
                self.config = config
                self.agent_executor = get_custom_agent(self.llm, self.config)
                self.original_system_msg = (self.system_msg+' ')[:-1]
                self.delete_scratchpad_logs()

        @property
        def system_msg(self)-> str:
                if self.config.agent_type.value == 'zeroshot react':
                        return self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template
                if self.config.agent_type.value == 'openai functioncall':
                        for runnable_comp in self.agent_executor.agent.runnable:
                                if runnable_comp[0]=='middle':
                                        return runnable_comp[1][0].messages[0].prompt.template
        

        def __call__(self, query):
                return (self.agent_executor.invoke({'input':query}))['output']


        def get_ruannble_comp(self, target:Literal['prompt', 'bound'])->Any:
                if self.config.agent_type.value == 'openai functioncall':
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
                        if self.config.agent_type.value == 'zeroshot react':
                                system_msg_breakdown = self.system_msg.split(' Format is Action')
                                appended_system_msg = system_msg_breakdown[0] + f" {msg} Format is Action" + system_msg_breakdown[1]
                                self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = appended_system_msg
                        if self.config.agent_type.value == 'openai functioncall':
                                appended_system_msg = self.system_msg + f' {msg}'  
                                self.get_ruannble_comp('prompt').messages[0].prompt.template = appended_system_msg
                        gr.Info("Appending system message succeeded!")
                        return self.system_msg                      
                except Exception as e:
                        raise gr.Error(e)

        def reset_system_msg(self)-> str:
                try:
                        if self.config.agent_type.value == 'zeroshot react':
                                self.agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = (self.original_system_msg+' ')[:-1]
                        if self.config.agent_type.value == 'openai functioncall':
                                self.get_ruannble_comp('prompt').messages[0].prompt.template = (self.original_system_msg+' ')[:-1]
                        gr.Info("Restoring system message succeeded!")
                        return self.system_msg
                except Exception as e:
                        raise gr.Error(e)




        def set_max_tokens(self, max_tokens:int)-> None:
                if self.config.agent_type.value == 'zeroshot react':
                        self.agent_executor.agent.llm_chain.llm.max_tokens = max_tokens
                if self.config.agent_type.value == 'openai functioncall':
                        self.get_ruannble_comp('bound').max_tokens = max_tokens


        def set_temperature(self, temperature: int)-> None:
                if self.config.agent_type.value == 'zeroshot react':
                        self.agent_executor.agent.llm_chain.llm.temperature = temperature
                if self.config.agent_type.value == 'openai functioncall':
                        self.get_ruannble_comp('bound').temperature = temperature        

        def delete_scratchpad_logs(self)-> None:
                try:
                        with open(os.path.join(self.config.scratchpad_log_folder, 'scratch_log.log'), 'w') as f:
                                pass
                except FileNotFoundError as e:
                        pass
