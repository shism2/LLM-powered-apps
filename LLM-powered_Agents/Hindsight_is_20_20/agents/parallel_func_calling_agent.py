import time, json, copy, ast, os
from typing import List, Type, Any, Optional,Literal, Tuple, Dict
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
from openai import RateLimitError
from langchain.adapters.openai import convert_message_to_dict
from langchain_core.runnables import RunnableLambda


def parsing_to_ai_msg_dict(chat_completion_msg):
    ai_msg_dict = {
        'role':'assistant',
        'content': chat_completion_msg.content,
        'tool_calls' : chat_completion_msg.tool_calls
    }
    return ai_msg_dict


def parsing_to_tool_msg_dict(tool_responses: List)-> List:
    result =   [{'role':'tool', 'name':tool_response[1], 'tool_call_id':tool_response[0], 'content':tool_response[2]} for tool_response in tool_responses]
    return result



class OpenAIParallelFuntionCallingAgent(BaseCustomAgent):
    def __init__(self, use_chat_completion_api:bool=False, azure_apenai_client: Optional=None, **kwargs):
        super().__init__(**kwargs)
        self.openai_functions = [convert_pydantic_to_openai_tool(x) for x in self.schemas]
        self.use_chat_completion_api = use_chat_completion_api
        self.azure_apenai_client = azure_apenai_client
        
        if self.use_chat_completion_api: 
            if self.azure_apenai_client==None:
                raise ValueError("If use_chat_completion_api, azure_apenai_client must be passed.")            
            self.messages.append({'role':'system', 'content':self.base_system_prompt.prompt.template})
            self.messages.append({'role':'user', 'content':''})
            self.parsing_to_ai_msg_dict_parser = parsing_to_ai_msg_dict
            self.parsing_to_tool_msg_dict_parser = parsing_to_tool_msg_dict


    @property
    def prompt(self)-> ChatPromptTemplate:
        raise NotImplementedError  

    @property
    def brain(self)-> RunnableSequence:  
        raise NotImplementedError      
        
    @retry(allowed_exceptions=(RateLimitError,))
    def _invoke_agent_action(self, query):
        if not self.use_chat_completion_api:
            raise NotImplementedError
        else:         
            self.messages[1]= {'role':'user', 'content':self.base_human_prompt.format_messages(input=query)[0].content}   
            # print(f"Size of messages: {len(self.messages)}")
            # print(self.messages)
            return self.parsing_to_ai_msg_dict_parser(
                        self.azure_apenai_client.chat_completions_create(
                        messages=self.messages, 
                        tools=self.openai_functions
                        )
                    )


    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        if not self.use_chat_completion_api:
            raise NotImplementedError
        else:      
            log = f'Unexpected Exception has been raised. The error message is "{e}"' if e != None else 'Unexpected Exception has been raised.'
            return {
                'role': 'assistant',
                'content' : log,
                'tool_calls' : None         
            }


    def run_agent_episode(self, query: str, reference: Optional[str]=None, trial: int=0, single_episode=False)-> None:    
        self._before_agent_episode(query=query, reference=reference)
        self.agent_log += f"Qurey: {query}\n"
        if single_episode:
            self.collect_logs(f"Trial {trial+1}", (True, 'info'), (True, 'info'), (True, 'info'))
            self.collect_logs(f"Query: {query}", (True, 'info'), (True, 'info'), (True, 'info'))
        else:
            self.collect_logs(f"Trial {trial+1}", (False, 'info'), (False, 'info'), (True, 'info'))
            self.collect_logs(f"Query: {query}", (False, 'info'), (False, 'info'), (True, 'info'))
        
        while not self.done:
            self.timestep += 1
            self.a, self.s_prime, done  = self.agent_step(query)       
            self.done = True if (done) or (self._is_halted(self.timestep)) else False

        # assessment the output
        if not self._is_halted(self.timestep):
            self.prediction = self.a['content'] 
        else: 
            self.prediction = "HALTED"
        self.judgement = self._assessment()
        self._add_judgement_to_agent_log()


    def _is_halted(self, timestep:int)-> bool:
        return self.timestep>self.horizon-1 and not isinstance(self.a, AgentFinish)



    def agent_step(self, query: str):
        """
        In Reinforcement-learning context, 'agent_step' method takes in 's' as input and returns 'a'.
        Here, the 'query', 'agent_action', and 'Observation' act as 's', 'a' and 's_prime', respectively.
        """
        self._before_agent_step()   
        if not self.use_chat_completion_api:
            raise NotImplementedError
        else:
            try:
                agent_action = self._invoke_agent_action(query)     
                self.messages.append(agent_action) # This must be here
                try:
                    Observation, temp_scratchpad = self._func_execution(agent_action=agent_action)
                except Exception as e:
                    print(f"agent_action is normal, but tool execution failed : {e}")
            except Exception as e:
                agent_action = self._invoke_agent_action_for_exception(e)
                self.messages.append(agent_action) # This must be here
                Observation, temp_scratchpad = self._func_execution_for_exception(e)
        self.agent_log += temp_scratchpad
        done = True if Observation[:8]=='Answer: ' else False 
        return agent_action, Observation, done       

    @retry(allowed_exceptions=(RateLimitError,), return_message=True)
    def _get_function_observation(self, tool, tool_input):
        return self.tool_dictionary[tool].run(**tool_input)

    # def _get_function_observation_2_(self, tool, tool_input):
    #     try:
    #         return self._get_function_observation(tool, tool_input)
    #     except Exception as e:
    #         return f"Unexpected Exception has been raised. The error message is {e}."

    def _func_execution(self, agent_action)-> Tuple[str,str]:
        '''
        In the Reinforcement-learning context, '_func_execution' method is reponsible for producing 'next state (s_prime)'.
        Here 'Obserbation' acts as 's_prime'. It additionally return the log string (Thought_Action+'\n'+Observation+'\n').  
        '''
        if not self.use_chat_completion_api:
            raise NotImplementedError

        tool_calls = agent_action['tool_calls']
        Observation_loglevel = 'error'
        Thought, Action = self._parsing_Thought_and_Action_into_str(tool_calls)        
        Thought_Action = Thought+'\n'+Action if Thought != '' else Action
        if tool_calls:
            try:
                observations = [(tool_call.id, tool_call.function.name, self._get_function_observation(tool_call.function.name, json.loads(tool_call.function.arguments))) for tool_call in tool_calls]
            except Exception as e:
                observations = [(tool_call.id, tool_call.function.name, f"Unexpected Exception has been raised. The error message is {e}.") for tool_call in tool_calls]
            tool_messages = self.parsing_to_tool_msg_dict_parser(observations)
            self.messages += tool_messages
            Observation = f"(step {self.timestep+1}-2) Results in parallel: [" + ', '.join( [f'Tool {i+1}-> '+(x['content'].strip()) for i, x in enumerate(tool_messages)]  ) + ']'
        else:
            Observation = f"Answer: {agent_action['content']}"
        Observation_loglevel = 'info'



        self.collect_logs(Observation, (True, Observation_loglevel), (True, Observation_loglevel), (True, Observation_loglevel))
        return Observation, Thought_Action+'\n'+Observation+'\n'



    def _func_execution_for_exception(self, e: Optional[str]=None) :              
        Action = f'(step {self.timestep+1}-1) {self.action_word[:-1]}: Could not invoke any tools because of the unexpected Exception. The error message is "{e}".' 
        self.collect_logs(Action, (True, 'error'), (True, 'error'), (True, 'error'))    
        
        Observation = f"(step {self.timestep+1}-2) Results in parallel: Could not get any tool-invocation results because of the unexpected Exception."
        self.collect_logs(Observation, (True, 'error'), (True, 'error'), (True, 'error'))    

        return Observation, Action+'\n'+Observation+'\n' 


    def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Override this property for any child class IF NECESSARY'''
        self.trial=0
        self.judgement = ['', "PENDING"]
        self.messages = self.messages[:2]



    def _parsing_Thought_and_Action_into_str(self, tool_calls:str)-> Tuple[str, str]:
        Action_loglevel = 'error'
        try:    
            Action = self._parsing_action_into_str(tool_calls)
            Action_loglevel = 'info'
        except Exception as e:
            Action = f'(step {self.timestep+1}-1) {self.action_word[:-1]}: Failed to parse Action into str. The error message is "{e}".'
        self.collect_logs(Action, (True, Action_loglevel), (True, Action_loglevel), (True, Action_loglevel))

        return '', Action 


    def __parse_into_func_name_args__(self, name, **kwargs):
        args_list = ', '.join([f"{key}={value}" for key, value in kwargs.items()])  
        return f"{name}({args_list})"          


    def _parsing_action_into_str(self, tool_calls:List[Dict])-> str:
        if not self.use_chat_completion_api:
            raise NotImplementedError

        try:
            if tool_calls:
                tools_and_args = [ {'name':tool_call.function.name, 'args':json.loads(tool_call.function.arguments)} for tool_call in tool_calls]
                results = [ self.__parse_into_func_name_args__(item['name'], **item['args']) for item in tools_and_args ]  
                Action = f'(step {self.timestep+1}-1) {self.action_word[:-1]}: [' + ', '.join([f'Tool {i+1}-> '+x for i, x in enumerate(results)]) + ']'
            else:
                Action = f'(step {self.timestep+1}-1) {self.action_word[:-1]}: Now I can answer directly without resorting to any tool.'

            return Action
        except Exception as e:
            raise Exception(e)
