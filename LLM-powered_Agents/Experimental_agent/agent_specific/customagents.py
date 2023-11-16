import os, sys
sys.path.extend(['..', '../..'])
import langchain
from langchain.agents import initialize_agent, AgentType, Tool
from typing import List, Any, Literal, Optional, Dict, Tuple
import gradio as gr
import copy
from utils.agent_components.get_llm import get_base_llm
from utils.agent_components.configurations import Configurations

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
from utils.agent_components.configurations import Configurations
from utils.agent_tools.tools.get_tools import get_tool_list

### For ReAct agent
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str




def get_custom_agent(llm, config:Configurations)-> langchain.agents.agent.AgentExecutor :

        from langchain.memory.utils import get_prompt_input_key
        class My_ConversationBufferMemory(ConversationBufferMemory):
                def _get_input_output(
                        self, inputs: Dict[str, Any], outputs: Dict[str, str]
                ) -> Tuple[str, str]:
                        if self.input_key is None:
                                prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
                        else:
                                prompt_input_key = self.input_key
                        if self.output_key is None:
                                output_key = list(outputs.keys())[0]
                        else:
                                output_key = self.output_key
                        return inputs[prompt_input_key], outputs[output_key]


        memory = My_ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        tool_result = get_tool_list(config)
        if isinstance(tool_result, tuple):
                tools = tool_result[0]
                tool_dictionary = tool_result[1]
        else:
                tools = tool_result

        try:
                agent_type = config.agent_type.value
        except AttributeError:
                agent_type = config.agent_type
        
        if agent_type == 'ReAct':       
                agent = get_ReAct_agents(llm=llm, tools=tools, RAG_style=False)
                system_msg_break_point = 'You have access to the following tools:'


        if agent_type == 'ReAct_RAG_style':         
                agent = get_ReAct_agents(llm=llm, tools=tools, RAG_style=True)
                system_msg_break_point = 'You have access to the following tools:'


        
        if agent_type == 'OpenAI_Functions':     
                agent = get_OpenAI_Functions_agent(llm=llm, tools=tools)
                system_msg_break_point = None
        
        '''
        My AgentExecutor
        '''
        from langchain.agents.agent import AgentExecutor
        from langchain.schema.agent import AgentFinish
        from langchain.callbacks.manager import CallbackManagerForChainRun        
        class My_AgentExecutor(AgentExecutor):
                def _return(
                        self,
                        output: AgentFinish,
                        intermediate_steps: list,
                        run_manager: Optional[CallbackManagerForChainRun] = None,
                ) -> Dict[str, Any]:
                        if run_manager:
                                run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
                        final_output = output.return_values
                        final_output["num_iterations"] = len(intermediate_steps)+1
                        if self.return_intermediate_steps:
                                final_output["output"] = intermediate_steps
                        return final_output

        agent_executor = My_AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        return agent_executor, system_msg_break_point



class ExperimentalAgent:
        def __init__(self, config:Configurations):
                self.config = config
                self.llm = get_base_llm(self.config)
                self.agent_executor, self.system_msg_break_point = get_custom_agent(self.llm, self.config)
                self.original_system_msg = (self.system_msg+' ')[:-1]
                self.delete_scratchpad_logs()

        @property
        def system_msg(self)-> str:
                for runnable_comp in self.agent_executor.agent.runnable:
                        if runnable_comp[0]=='middle':
                                return runnable_comp[1][0].messages[0].prompt.template
        

        def __call__(self, query, return_iter_num = False)-> str:
                try: agent_type = self.config.agent_type.value
                except AttributeError: agent_type = self.config.agent_type
                response = self.agent_executor.invoke({'input':query}, {"metadata": {"agent_type": agent_type}})
                if not return_iter_num:
                        return response['output']
                else:
                        return response['output'], response['num_iterations']
                


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
                        gr.Info("Resetting system message succeeded!")
                        return self.system_msg
                except Exception as e:
                        raise gr.Error(e)




        def set_max_tokens(self, max_tokens:int|None)-> None:
                self.get_ruannble_comp('llm').max_tokens = max_tokens


        def set_temperature(self, temperature: int)-> None:
                self.get_ruannble_comp('llm').temperature = temperature        

        def delete_scratchpad_logs(self)-> None:
                try:
                        with open(os.path.join(self.config.scratchpad_log_folder, 'scratch_log.log'), 'w') as f:
                                pass
                except FileNotFoundError as e:
                        pass

        @classmethod
        def from_kwargs(cls, **kwargs):
                config = Configurations(**kwargs)
                return cls(config)

