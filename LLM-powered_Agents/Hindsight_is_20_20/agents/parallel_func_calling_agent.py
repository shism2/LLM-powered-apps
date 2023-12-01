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
from reasoning_engines.langchain_llm_wrappers import AsyncQuickAzureOpenAIClient
import asyncio



def parsing_to_ai_msg_dict(chat_completion_msg):
    return {'role':'assistant', 'content': chat_completion_msg.content, 'tool_calls' : chat_completion_msg.tool_calls}

async def parsing_to_ai_msg_dict_a(chat_completion_msg):
    return parsing_to_ai_msg_dict(chat_completion_msg)

async def parsing_to_ai_msg_dict_async(chat_completion_msg):
    ai_msg_dict = await parsing_to_ai_msg_dict_a(chat_completion_msg)
    return ai_msg_dict

def parsing_to_tool_msg_dict(tool_responses: List)-> List:
    return [{'role':'tool', 'name':tool_response[1], 'tool_call_id':tool_response[0], 'content':tool_response[2]} for tool_response in tool_responses]

async def parsing_to_tool_msg_dict_a(tool_responses: List)-> List:
    return parsing_to_tool_msg_dict(tool_responses)

async def parsing_to_tool_msg_dict_async(tool_responses: List)-> List:
    result =  await parsing_to_tool_msg_dict_a(tool_responses=tool_responses) 
    return result



class OpenAIParallelFuntionCallingAgent(BaseCustomAgent):
    @property
    def is_reflexion_agent(self):
        return False

    @property
    def prompt(self)-> ChatPromptTemplate:
        raise NotImplementedError  

    @property
    def brain(self)-> RunnableSequence:  
        raise NotImplementedError 


    def __init__(self, use_chat_completion_api:bool=False, azure_apenai_client: Optional=None, **kwargs):        
        self.use_chat_completion_api = use_chat_completion_api
        super().__init__(**kwargs)
        self.openai_functions = [convert_pydantic_to_openai_tool(x) for x in self.schemas]
        self.azure_apenai_client = azure_apenai_client
        if not isinstance(self.azure_apenai_client, AsyncQuickAzureOpenAIClient):
            self.collect_logs("You passed in a synchronous Azure client. It is recommended you use an Async client.", (True, 'error'), (False, 'error'), (False, 'error'))
            raise Exception
        
        if self.use_chat_completion_api: 
            if self.azure_apenai_client==None:
                raise ValueError("If use_chat_completion_api, azure_apenai_client must be passed.")
            self.parsing_to_ai_msg_dict_parser = parsing_to_ai_msg_dict
            self.parsing_to_ai_msg_dict_parser_async = parsing_to_ai_msg_dict_async
            self.parsing_to_tool_msg_dict_parser = parsing_to_tool_msg_dict
            self.parsing_to_tool_msg_dict_parser_async = parsing_to_tool_msg_dict_async

    #######################################################################

    async def append_messages_async(self, message):
        self.messages.append(message)

    async def populate_user_message_async(self, query):
        self.messages[1]= {'role':'user', 'content':self.base_human_prompt.format_messages(input=query)[0].content}  


    @retry(allowed_exceptions=(RateLimitError,))
    async def _invoke_agent_action_async(self, query):
        if not self.use_chat_completion_api:
            raise NotImplementedError
                 
        await self.populate_user_message_async(query)
        chat_response = await self.azure_apenai_client.chat_completions_create(messages=self.messages, tools=self.openai_functions)
        chat_response = await self.parsing_to_ai_msg_dict_parser_async(chat_response)                            
        return chat_response 

    ### _invoke_agent_action_for_exception ### 
    async def _invoke_agent_action_for_exception_async(self, e: Optional[str]=None):
        if not self.use_chat_completion_api:
            raise NotImplementedError
        
        log = f'Unexpected Exception has been raised. The error message is "{e}"' if e != None else 'Unexpected Exception has been raised.'
        return {'role': 'assistant', 'content' : log, 'tool_calls' : None}



    async def run_agent_episode_async(self, query: str, reference: Optional[str]=None, trial: int=0, single_episode=True)-> None:    
        await self._before_agent_episode_async(query=query, reference=reference)
        self.agent_log += f"Qurey: {query}\n"
        if single_episode:
            await self.collect_logs_async(f"----- New single test point -----", (False, 'info'), (True, 'info'), (True, 'info'))
            await self.collect_logs_async(f"Query: {query}", (True, 'info'), (True, 'info'), (True, 'info'))
        else:
            await self.collect_logs_async(f"Trial {trial+1}", (False, 'info'), (False, 'info'), (True, 'info'))
            await self.collect_logs_async(f"Query: {query}", (False, 'info'), (False, 'info'), (True, 'info'))
        
        while not self.done:
            self.a, self.s_prime, self.is_termination_state  = await self.agent_step_async(query)       
            self.done = True if (self.is_termination_state) or (self._is_halted(self.timestep)) else False

            if not self.done:
                self.timestep += 1

        # assessment the output
        if not self._is_halted(self.timestep):
            self.prediction = self.a['content'] 
        else: 
            self.prediction = "HALTED"
        self.judgement = await self._assessment_async()
        await self._add_judgement_to_agent_log_async()



    async def run_agent_trials_async(self, num_trials: int, query: str, reference: Optional[str]=None)-> None:
        await self._before_agent_trials_async(query=query, reference=reference)
        await self.collect_logs_async(f"----- New test point -----", (False, 'info'), (True, 'info'), (True, 'info'))
        await self.collect_logs_async(f"Query: {query}", (True, 'info'), (True, 'info'), (False, 'info'))
        
        while self.judgement[1]!='CORRECT' and self.trial<num_trials:  
            await self.collect_logs_async(f"Trial {self.trial+1}", (True, 'info'), (True, 'info'), (False, 'info'))
            await self.run_agent_episode_async(query=query, reference=reference, trial=self.trial, single_episode=False)  
            self.trial += 1




    async def agent_step_async(self, query: str):
        """
        In Reinforcement-learning context, 'agent_step' method takes in 's' as input and returns 'a'.
        Here, the 'query', 'agent_action', and 'Observation' act as 's', 'a' and 's_prime', respectively.
        """
        self._before_agent_step()   
        if not self.use_chat_completion_api:
            raise NotImplementedError
        
        try:
            agent_action = await self._invoke_agent_action_async(query)
            await self.append_messages_async(agent_action) # This must be here
        except Exception as e:
            agent_action = await self._invoke_agent_action_for_exception_async(e)
            await self.append_messages_async(agent_action) # This must be here
            Observation, temp_scratchpad = await self._func_execution_for_exception_async(e)
        
        Observation, temp_scratchpad = await self._func_execution_async(agent_action=agent_action)
        
        
        self.agent_log += temp_scratchpad
        done = True if Observation[:8]=='Answer: ' else False 
        return agent_action, Observation, done       

    @retry(allowed_exceptions=(RateLimitError,), return_message="Successfully got a tool to invoke but could not invoke it due to Exception.")
    def _get_function_observation(self, tool, tool_input):
        return self.tool_dictionary[tool].run(**tool_input)

    async def _get_function_observation_a(self, tool, tool_input):
        return self._get_function_observation(tool=tool, tool_input=tool_input)

    async def _get_function_observation_async(self, tool, tool_input):
        invocation_result = await self._get_function_observation_a(tool=tool, tool_input=tool_input)
        return invocation_result


    async def _func_execution_async(self, agent_action)-> Tuple[str,str]:
        if not self.use_chat_completion_api:
            raise NotImplementedError

        tool_calls = agent_action['tool_calls']
        Observation_loglevel = 'error'
        Thought, Action = await self._parsing_Thought_and_Action_into_str_async(tool_calls)
        Thought_Action = Thought+'\n'+Action if Thought != '' else Action
        if tool_calls:
            
            tasks = await asyncio.gather(*[self._get_function_observation_async(tool_call.function.name, json.loads(tool_call.function.arguments)) for tool_call in tool_calls])
            observations = [(tool_call.id, tool_call.function.name, task) for task, tool_call in zip(tasks, tool_calls)]
            tool_messages = await self.parsing_to_tool_msg_dict_parser_async(observations)     
            self.messages += tool_messages

            are_all_tools_excpetion = await self.are_all_tools_excpetion(observations)
            if not are_all_tools_excpetion:
                Observation_loglevel = 'info'
                Observation = f"(step {self.timestep+1}-observation) {self.observation_word[-1]}: [" + ', '.join( [f'Tool {i+1}-> '+(x['content'].strip()) for i, x in enumerate(tool_messages)]  ) + ']'
            else: 
                Observation = f'(step {self.timestep+1}-observation) {self.observation_word[-1]}: Successfully got a list of tools to invoke but could not invoke them due to Exception.'
        else:
            Observation_loglevel = 'info'
            Observation = f"Answer: {(agent_action['content']).strip()}" 

        await self.collect_logs_async(Observation, (True, Observation_loglevel), (True, Observation_loglevel), (True, Observation_loglevel))  
        return Observation, Thought_Action+'\n'+Observation+'\n'

    async def are_all_tools_excpetion(self, observations):
        return sum([ self.contains_word(x[2], 'Exception') for x in observations]) == len(observations)


    #### _func_execution_for_exception #### 
    def _func_execution_for_exception(self, e: Optional[str]=None, collect_logs:Tuple[bool, bool]=(True, True)) :              
        Action = f'(step {self.timestep+1}-action) {self.action_word[:-1]}: Could not get any response from the Brain (LLM) due to Exception. The error message is "{e}".' 
        if collect_logs[0]:
            self.collect_logs(Action, (True, 'error'), (True, 'error'), (True, 'error'))
        Observation = f'(step {self.timestep+1}-observation) {self.observation_word[-1]}: Could not get any list of tool to invoke due to Exception. The error message is "{e}".'
        if collect_logs[1]:
            self.collect_logs(Observation, (True, 'error'), (True, 'error'), (True, 'error'))
        return Observation, Action+'\n'+Observation+'\n'
    
    async def _func_execution_for_exception_a(self, e: Optional[str]=None, collect_logs:Tuple[bool, bool]=(True, True)):      
        return self._func_execution_for_exception(e, collect_logs=collect_logs)

    async def _func_execution_for_exception_async(self, e: Optional[str]=None, collect_logs:Tuple[bool, bool]=(True, True)) :    
        Observation, temp_scratchpad = await self._func_execution_for_exception_a(e, collect_logs=collect_logs)
        return Observation, temp_scratchpad 





    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        super()._before_agent_episode(query=query, reference=reference)
        if self.use_chat_completion_api:
            self.messages = []
            self.messages.append({'role':'system', 'content':self.base_system_prompt.prompt.template})
            self.messages.append({'role':'user', 'content':''})

    async def _before_agent_episode_a(self, query: Optional[str]=None, reference: Optional[str]=None):
        self._before_agent_episode(query=query, reference=reference)


    async def _before_agent_episode_async(self, query: Optional[str]=None, reference: Optional[str]=None):
        await self._before_agent_episode_a(query=query, reference=reference)




    def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
        super()._before_agent_trials(query=query, reference=reference)
        if self.use_chat_completion_api:
            self.messages = []
            self.messages.append({'role':'system', 'content':self.base_system_prompt.prompt.template})
            self.messages.append({'role':'user', 'content':''})

    async def _before_agent_trials_a(self, query: Optional[str]=None, reference: Optional[str]=None):
        self._before_agent_trials(query=query, reference=reference)
    
    async def _before_agent_trials_async(self, query: Optional[str]=None, reference: Optional[str]=None):
        await self._before_agent_trials_a(query=query, reference=reference)



    #### _parsing_Thought_and_Action_into_str #### 
    def _parsing_Thought_and_Action_into_str(self, tool_calls:str, collect_logs=True)-> Tuple[str, str]:
        Action_loglevel = 'error'
        try:    
            Action = self._parsing_action_into_str(tool_calls)
            Action_loglevel = 'info'
        except Exception as e:
            Action = f'(step {self.timestep+1}-action) {self.action_word[:-1]}: Successfully got a list of tools to invoke but could not parse them into string format due to parsing error. The error message is "{e}".'
        if collect_logs:
            self.collect_logs(Action, (True, Action_loglevel), (True, Action_loglevel), (True, Action_loglevel))
        
        return '', Action 

    async def _parsing_Thought_and_Action_into_str_a(self, tool_calls:str, collect_logs=True)-> Tuple[str, str]:   
        _, Action = self._parsing_Thought_and_Action_into_str(tool_calls, collect_logs=collect_logs) 
        return '', Action
    
    async def _parsing_Thought_and_Action_into_str_async(self, tool_calls:str, collect_logs=True)-> Tuple[str, str]:
        _, Action = await self._parsing_Thought_and_Action_into_str_a(tool_calls, collect_logs=collect_logs)
        
        return '', Action 



    #### _parsing_action_into_str #### 
    def _parsing_action_into_str(self, tool_calls:List[Dict])-> str:
        if not self.use_chat_completion_api:
            raise NotImplementedError
        try:
            if tool_calls:
                tools_and_args = [ {'name':tool_call.function.name, 'args':json.loads(tool_call.function.arguments)} for tool_call in tool_calls]
                results = [ self.__parse_into_func_name_args__(item['name'], **item['args']) for item in tools_and_args ]  
                Action = f'(step {self.timestep+1}-action) {self.action_word[:-1]}: [' + ', '.join([f'Tool {i+1}-> '+x for i, x in enumerate(results)]) + ']'
            else:
                Action = f'(step {self.timestep+1}-action) {self.action_word[:-1]}: Now I can answer directly without resorting to any tool.'
            return Action
        except Exception as e:
            raise Exception(e)

    def __parse_into_func_name_args__(self, name, **kwargs):
        args_list = ', '.join([f"{key}={value}" for key, value in kwargs.items()])  
        return f"{name}({args_list})"          