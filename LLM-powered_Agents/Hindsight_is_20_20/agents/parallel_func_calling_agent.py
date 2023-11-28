import time, json, copy, ast, os
from typing import List, Type, Any, Optional,Literal, Tuple
from pydantic import BaseModel, Field
from agents.base_cumtom_agent import BaseCustomAgent

from langchain.schema.agent import AgentAction, AgentFinish, AgentActionMessageLog
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from utils.wrappers import retry


class MyAIMessageToAgentActionParserForParallelFunctionCalling(JsonOutputToolsParser):
    tools: List[Type[BaseModel]]    
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:

        generation = result[0] if isinstance(result, list) else result
        if generation.text:
            log = generation.text
            return_values = {'output':log}
            return AgentFinish(return_values=return_values, log=log, type='AgentFinish')
        else:
            generation = generation.message
            tool_calls = copy.deepcopy(generation.additional_kwargs["tool_calls"])
            results = []
            for tool_call in tool_calls:
                if "function" not in tool_call:
                    pass
                function_args = tool_call["function"]["arguments"]
                results.append(
                    {
                        "type": tool_call["function"]["name"],
                        "args": json.loads(function_args),
                    }
                )
            # print(result)
            tool = [res["type"] for res in results]
            name_dict = {tool.__name__: tool for tool in self.tools}
            
            args = [name_dict[res["type"]](**res["args"]) for res in results]
            log = '\nInvoking: ['+', '.join([x+'('+str(y)+')' for x, y in zip(tool, args)])+']'
            return AgentActionMessageLog(log=log, tool=', '.join(tool), tool_input={'tool_input': [(res["type"],res["args"]) for res in results] }, message_log=[generation])

class OpenAIParallelFuntionCallingAgent(BaseCustomAgent):
    def __init__(self, **kwargs):
        """tools: List[Tuple[Type[schema], Dict[str, Tool]]] """
        super().__init__(**kwargs)
        self.function_descriptions = [convert_pydantic_to_openai_tool(x) for x in self.schemas]
        self.parser = MyAIMessageToAgentActionParserForParallelFunctionCalling(tools=self.schemas)

    @property
    def prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
                self.base_system_prompt,
                self.base_human_prompt,
                # MessagesPlaceholder(variable_name="agent_scratchpad"), 
            ])
        return prompt

    @property
    def brain(self)-> RunnableSequence:        
        return  (
            RunnablePassthrough.assign(agent_scratchpad = lambda x : format_to_openai_function_messages(x['intermediate_steps']))
            | self.prompt 
            # | self.reasoninig_engine.bind(tools=self.function_descriptions, tool_choice='auto')
            | self.reasoninig_engine
            # | self.parser
        )


    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
        return AgentActionMessageLog(
                log=log,
                tool='',
                tool_input='',
                type = 'AgentActionMessageLog')

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


    def _func_execution(self, agent_action: AgentAction|AgentActionMessageLog|AgentFinish)-> Tuple[str,str]:
        '''
        In the Reinforcement-learning context, '_func_execution' method is reponsible for producing 'next state (s_prime)'.
        Here 'Obserbation' acts as 's_prime'. It additionally return the log string (Thought_Action+'\n'+Observation+'\n').  
        '''

        Thought, Action = self._parsing_Thought_and_Action_into_str(agent_action.log)
        Thought_Action = Thought+'\n'+Action if Thought != '' else Action

        # Either Observation or Answer
        Observation_loglevel = 'error'
        if isinstance(agent_action, AgentAction):
            try:         
                tool_batch = agent_action.tool_input['tool_input']
                Observation_batch = [self._get_function_observation(tool, tool_input) for tool, tool_input in tool_batch]
                Observation = (f'Observation {self.timestep+1}: {str(Observation_batch)}').rstrip('\n')
                Observation_loglevel = 'info'
            except Exception as e:
                Observation = f'Observation {self.timestep+1}: Failed to get Observation (function output). The tool is {agent_action.tool} and tool input is {agent_action.tool_input}. The error message is "{e}"'
        else:
            try:
                Observation = (f'Answer: '+agent_action.return_values['output']).rstrip('\n') 
                Observation_loglevel = 'info'
            except Exception as e:
                Observation = f'Answer: Failed to get the final answer. The error message is "{e}"'
        
        self.collect_logs(Observation, (True, Observation_loglevel), (True, Observation_loglevel), (True, Observation_loglevel))
        return Observation, Thought_Action+'\n'+Observation+'\n'
