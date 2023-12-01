import pytz, re, copy, json
from typing import Optional, List, Dict, Any, Tuple, Literal, Type
from agents.base_cumtom_agent import BaseCustomAgent
from judgement.criteria import QA_Evaluator
from datetime import datetime
from utils.agent_tools.tools.get_tools import get_tool_list

from pydantic import BaseModel, Field
from langchain_core.outputs import ChatGeneration, Generation
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableSequence
from langchain.schema.agent import AgentAction, AgentFinish, AgentActionMessageLog
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.output_parsers.openai_tools import JsonOutputToolsParser

class MyAIMessageToAgentActionParser(JsonOutputToolsParser):
    tools: List[Type[BaseModel]]   

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0].message if isinstance(result, List) else result.message
        if generation.content:
            log = generation.content
            return_values = {'output':log}
            return AgentFinish(return_values=return_values, log=log, type='AgentFinish')
        else:
            function_call = copy.deepcopy(generation.additional_kwargs["function_call"])
            function_args = function_call["arguments"]
            function = {
                    "name": function_call["name"],
                    "args": json.loads(function_args),
                }

            name_dict = {tool.__name__: tool for tool in self.tools}
            args = name_dict[function["name"]](**function["args"])
            log = f'\nInvoking: ' + function["name"]+ '('+ str(args) + ')'
            return AgentActionMessageLog(log=log, tool=function["name"], tool_input=function["args"], message_log=[generation])


class OpenAIFuntionCallingAgent(BaseCustomAgent):
    @property
    def is_reflexion_agent(self):
        return False

    @property
    def prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
                self.base_system_prompt,
                self.base_human_prompt,
                MessagesPlaceholder(variable_name="agent_scratchpad"), # List of messages
            ])
        return prompt

    @property
    def brain(self)-> RunnableSequence:        
        return  (
            RunnablePassthrough.assign(agent_scratchpad = lambda x : format_to_openai_function_messages(x['intermediate_steps'])) # Get a list of messages
            | self.prompt 
            | self.reasoninig_engine.bind(functions=self.openai_functions)
            | MyAIMessageToAgentActionParser(tools=self.schemas)
        )


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.openai_functions = [format_tool_to_openai_function(f) for f in self.tools]


    ''' <<< Invoke Brain >>> '''
    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
        return AgentActionMessageLog(
                log=log,
                tool='',
                tool_input='',
                type = 'AgentActionMessageLog')
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< Parsing into string >>>'''
    def _parsing_action_into_str(self, raw_action_string:str)-> str:
        try:
            if self.contains_word(raw_action_string, self.action_word):
                # AgentAction
                action = raw_action_string.split(self.action_word+' ')[-1]
                Action = f"{self.action_word[:-1]} {self.timestep+1}: {action}"
            else:
                # AgentFinish
                Action = f"{self.action_word[:-1]} {self.timestep+1}: {raw_action_string}"
        except Exception as e:
            raise Exception(e)
        return Action
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
