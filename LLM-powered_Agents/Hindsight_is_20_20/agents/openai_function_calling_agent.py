import pytz, re
from typing import Optional, List, Dict, Any, Tuple, Literal
from agents.base_cumtom_agent import BaseCustomAgent
from judgement.criteria import QA_Evaluator
from datetime import datetime
from utils.agent_tools.tools.get_tools import get_tool_list

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableSequence
from langchain.schema.agent import AgentAction, AgentFinish, AgentActionMessageLog
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function

class OpenAIFuntionCallingAgent(BaseCustomAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.openai_functions = [format_tool_to_openai_function(f) for f in self.tools]

    @property
    def prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
                self.base_system_prompt,
                self.base_human_prompt,
                MessagesPlaceholder(variable_name="agent_scratchpad"), 
            ])
        return prompt

    @property
    def brain(self)-> RunnableSequence:        
        return  (
            RunnablePassthrough.assign(agent_scratchpad = lambda x : format_to_openai_functions(x['intermediate_steps']))
            | self.prompt 
            | self.reasoninig_engine.bind(functions=self.openai_functions)
            | OpenAIFunctionsAgentOutputParser()
        )



    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        ''' Override this property for any child class  IF NECESSARY'''
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
        return AgentAction(
                log=log,
                tool='Error!! Excetion has been raised.',
                tool_input='Error!! Excetion has been raised.',
                type = 'AgentAction')


    def _parsing_action_into_str(self, raw_action_string:str)-> str:
        function_name_match = re.search(r"`(.*?)`", raw_action_string)  ## Can be None
        arguments_match = re.search(r"\{(.*?)\}", raw_action_string)  ## Can be None
        if function_name_match and arguments_match:  
            # agent action is not finished and function argument is json blob  
            function_name = function_name_match.group(1)  
            arguments = arguments_match.group(1)  

            processed_arguments = []  
            for arg in arguments.split(', '):
                key, value = arg.split(': ')    
                key = key.strip("'")  
                value = value.strip("'") if not (value.startswith("'") and value.endswith("'")) else value   
                processed_arguments.append(f"{key}={value}")                  
            arguments_str = ', '.join(processed_arguments)  
            Action = f"{self.action_word[:-1]} {self.timestep+1}: {function_name}({arguments_str})"  
        
        elif function_name_match and arguments_match==None:
            # agent action is not finished and function argument is NOT json blob
            function_name = function_name_match.group(1)  
            try:
                arguments_str = (raw_action_string.split(function_name)[-1].strip()).split('`')[-2]
                Action = f"{self.action_word[:-1]} {self.timestep+1}: {function_name}({arguments_str})"  
            except Exception as e:
                raise Exception(e)
        else:
            # agent action is finished
            Action = f"{self.action_word[:-1]} {self.timestep+1}: {raw_action_string}"

        return Action

####################################################################
if __name__ == '__main__':
    from utils.agent_components.get_llm import LangChainLLMWrapper

    openai_agent = OpenAIFuntionCallingAgent(
        reasoninig_engine = LangChainLLMWrapper().llm,
        base_prompt = "jet-taekyo-lee/openai-function-calling-agent",
        tools = get_tool_list(),
        evaluator = QA_Evaluator(),
        action_word = 'Invoking'
    )

