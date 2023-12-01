import os, copy, json, re, pytz, logging  
from typing import List, Tuple, Any, Dict, Optional, Literal, Type
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import Tool
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from loggers.get_loggers import get_hd22_file_logger, get_hd22_stream_logger
from openai import RateLimitError
import time
from utils.wrappers import retry
from abc import ABC, abstractmethod




class BaseCustomAgent:
    @property
    def is_reflexion_agent(self):
        return False

    @property
    def base_prompt(self)-> ChatPromptTemplate:
        ''' Override this property for any child class '''
        raise NotImplementedError

    @base_prompt.setter
    def base_prompt(self, new_prompt: ChatPromptTemplate)-> None:
        ''' Override this property for any child class '''
        raise NotImplementedError    

    @property
    def brain(self)-> Any:
        ''' Override this property for any child class '''
        raise NotImplementedError


    def __init__(self,
                reasoninig_engine: Any,
                base_prompt: ChatPromptTemplate|str,
                schemas_and_tools: List[Tuple[BaseModel, Dict[str, Tool]]],
                evaluator: Any,
                action_word: str,
                thought_word: Optional[str]=None, 
                observation_word: Optional[str]=None, 
                stop_words: List[str]|None=None,
                retry_RateLimitError: bool = True,
                retry_standby = 20,
                horizon: int=5,
                e2e_log_folder: str='loggers/e2e_logs',
                trajectory_only_log_folder: str='loggers/trajectory_only_logs',
                file_logging: bool=True,
                colsole_logging: bool=True,
                )-> None:
        
        ### Mapping argumnets into attributes
        self.reasoninig_engine: Any = reasoninig_engine
        if isinstance(base_prompt, str):
            base_prompt = hub.pull(base_prompt)
        if len(base_prompt)!=2 or not isinstance(base_prompt[0], SystemMessagePromptTemplate) or not isinstance(base_prompt[1], HumanMessagePromptTemplate): 
            raise ValueError("Error in 'base_prompt'.")
        self.base_system_prompt = base_prompt[0]
        self.base_human_prompt = base_prompt[1]        
        self.schemas_and_tools = schemas_and_tools
        self.evaluator = evaluator
        self.action_word = action_word+':' if (action_word!=None and len(action_word.split(':'))==1) else action_word
        self.thought_word = thought_word+':' if (thought_word!=None and len(thought_word.split(':'))==1) else thought_word
        self.observation_word = observation_word+':' if (observation_word!=None and len(observation_word.split(':'))==1) else observation_word
        self.retry_RateLimitError = retry_RateLimitError
        self.retry_standby = retry_standby
        self.stop_words = stop_words
        self.horizon = horizon
        self.e2e_log_folder = e2e_log_folder
        self.trajectory_only_log_folder = trajectory_only_log_folder
        self.file_logging = file_logging
        self.colsole_logging = colsole_logging



        ### Induced attributes
        self.schemas = self.schemas_and_tools.schemas()
        self.tools = self.schemas_and_tools.tools()
        self.tool_dictionary = self.schemas_and_tools.tool_dictionary()
        self.e2e_logger = get_hd22_file_logger(log_file= os.path.join(self.e2e_log_folder, self.get_logger_name_prefix()+'_e2e.log'), logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_e2e' )
        self.trajectory_logger = get_hd22_file_logger(log_file= os.path.join(self.trajectory_only_log_folder, self.get_logger_name_prefix()+'_trajectory.log'), logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_trajectory' )
        self.console_logger = get_hd22_stream_logger(logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_console')
        self.messages = [] # For chat-completion api use



        ### Intricsic parameters
        self.timestep = 0
        self.query: str = ''
        self.reference: str = ''
        self.done: bool = False
        self.is_termination_state: bool = False
        self.prediction: str = ''
        self.judgement: List = ['', 0]
        self.agent_log: str = ''
        self.trajectory_only_log_for_reflexion: str = ''
        self.trial: int =0
        self.a = None
        self.s_prime = ''

        # agent_reset
        self._before_agent_episode()
        self._before_agent_trials(reference='Instance initialization')


<<<<<<< HEAD


    ####################### Fundamental methods #######################
=======
>>>>>>> temp_1125
    def agent_reset(self):
        self._before_agent_episode()
        self._before_agent_trials(reference='Instance initialization')

    def clear_logs(self):
        with open(os.path.join(self.e2e_log_folder, self.get_logger_name_prefix()+'_e2e.log'), 'w') as f:
            f.write('')
        with open(os.path.join(self.trajectory_only_log_folder, self.get_logger_name_prefix()+'_trajectory.log'), 'w') as f:
            f.write('')

    def collect_logs(self, message: str, console: Tuple[bool, str], e2e: Tuple[bool, str], trajectory: Tuple[bool, str]):
        colsole_log_method = getattr(self.console_logger, console[1].lower())
        if self.colsole_logging and console[0]:
            colsole_log_method(message)         

        e2e_log_method = getattr(self.e2e_logger, e2e[1].lower())  
        if self.file_logging and e2e[0]:
            e2e_log_method(message)
        
        trajectory_log_method = getattr(self.trajectory_logger, trajectory[1].lower())         
        if self.file_logging and trajectory[0]:
            trajectory_log_method(message)

    async def collect_logs_a(self, message: str, console: Tuple[bool, str], e2e: Tuple[bool, str], trajectory: Tuple[bool, str]):
        self.collect_logs(message=message, console=console, e2e=e2e, trajectory=trajectory)
    
    async def collect_logs_async(self, message: str, console: Tuple[bool, str], e2e: Tuple[bool, str], trajectory: Tuple[bool, str]):
        await self.collect_logs_a(message=message, console=console, e2e=e2e, trajectory=trajectory)

    def get_logger_name_prefix(self):
        korea_time = datetime.now(pytz.timezone('Asia/Seoul'))  
        year, month, day = korea_time.year, korea_time.month, korea_time.day
        return f'{self.__class__.__name__}_{year}-{month}-{day}'

    def contains_word(self, sentence, word):  
        words = sentence.lower().split()    
        return word.lower() in words  



    ''' <<< Invoke Brain >>> '''
    @retry(allowed_exceptions=(RateLimitError,))
    def _invoke_agent_action(self, query):
        ''' Override this property for any child class  IF NECESSARY'''
        return self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query
            })

    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        ''' Suppoer Acync by appending _async '''
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
        log += '\nAction:\n```\n{\n"action": "",\n"action_input": ""\n}\n```'
        return AgentAction(
                log=log,
                tool='',
                tool_input='',
                type = 'AgentAction')
    
    async def _invoke_agent_action_for_exception_a(self, e: Optional[str]=None):
        return self._invoke_agent_action_for_exception(e)
    
    async def _invoke_agent_action_for_exception_async(self, e: Optional[str]=None):
        result = await self._invoke_agent_action_for_exception_a(e)
        return result
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< run_agent_step >>> '''
    def _before_agent_step(self):
        ''' Override this property for any child class IF NECESSARY'''
        pass


    def agent_step(self, query: str):
        """
        In Reinforcement-learning context, 'agent_step' method takes in 's' as input and returns 'a'.
        Here, the 'query', 'agent_action', and 'Observation' act as 's', 'a' and 's_prime', respectively.
        """
        self._before_agent_step()        
        try: 
            ''' Invoke Brain (LLM) '''
            agent_action = self._invoke_agent_action(query)        
        except Exception as e:
            """ Exception to Brain (LLM) """
            agent_action = self._invoke_agent_action_for_exception(e)
            Observation, temp_scratchpad = self._func_execution_for_exception(e)
        
        ''' No Exception to Brain (LLM) '''
        Observation, temp_scratchpad = self._func_execution(agent_action=agent_action)
        
        self.agent_log += temp_scratchpad
        is_termination_state = True if isinstance(agent_action, AgentFinish) else False 
        return agent_action, Observation, is_termination_state       
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< tool invocation >>> '''
    @retry(allowed_exceptions=(RateLimitError,))
    def _get_function_observation(self, tool, tool_input):
        ''' Suppost Async by appending _async'''
        return self.tool_dictionary[tool].run(**tool_input)

    async def _get_function_observation_a(self, tool, tool_input):
        return self._get_function_observation(tool=tool, tool_input=tool_input)

    async def _get_function_observation_async(self, tool, tool_input):
        invocation_result = await self._get_function_observation_a(tool=tool, tool_input=tool_input)
        return invocation_result

    def _func_execution(self, agent_action: AgentAction|AgentFinish|Literal['NO_NEED'])-> Tuple[str,str]:
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
                observation = self._get_function_observation(agent_action.tool, agent_action.tool_input)
                Observation = (f'Observation {self.timestep+1}: '+observation).rstrip('\n')
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



    def _func_execution_for_exception(self, e: Optional[str]=None) :        
        ''' Suppost Async by appending _async'''
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
         
        Thought = f'{self.thought_word[:-1]} {self.timestep+1}: '+log if self.thought_word!=None else ''
        if self.thought_word!=None:
            self.collect_logs(Thought, (True, 'error'), (True, 'error'), (True, 'error'))    
        
        Action = f'{self.action_word[:-1]} {self.timestep+1}: '+ 'Could not get an Action because Exception has been raised.'
        if self.thought_word == None:
            Action += f'The error message is "{e}"'
        self.collect_logs(Action, (True, 'error'), (True, 'error'), (True, 'error'))    
        
        Observation = f'Observation {self.timestep+1}: '+ 'Could not get an Observation because Exception has been raised.'
        self.collect_logs(Observation, (True, 'error'), (True, 'error'), (True, 'error'))    

        return Observation, Thought+'\n'+Action+'\n'+Observation+'\n' 

    async def _func_execution_for_exception_a(self, e: Optional[str]=None, collect_logs:Tuple[bool, bool]=(True, True)):      
        return self._func_execution_for_exception(e, collect_logs=collect_logs)

    async def _func_execution_for_exception_async(self, e: Optional[str]=None, collect_logs:Tuple[bool, bool]=(True, True)) :    
        Observation, temp_scratchpad = await self._func_execution_for_exception_a(e, collect_logs=collect_logs)
        return Observation, temp_scratchpad 

    async def _are_all_tools_excpetion(self, observations:List):
        return sum([ self.contains_word(x[2], 'Exception') for x in observations]) == len(observations)        
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



    ''' <<< run_agent_episode >>> '''
    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Support Async by appending _async '''
        self.query = query
        self.reference = reference if reference != None else ''
        self.agent_log = ''
        self.prediction = ''
        self.timestep = 0
        self.done  = False   
        self.is_termination_state = False
        self.intermediate_steps = []
        self.judgement = ['', "PENDING"]
        self.a = None
        self.s_prime = ''
        if self.is_reflexion_agent:
            if self.trial>0:
                    self.collect_logs(f"Reflexion......", (True, 'info'), (True, 'info'), (False, 'info'))            
                    self._do_reflexion_(self.trajectory_only_log_for_reflexion)
                    reflexion_loglevel = 'info' if len(self.most_recent_reflexion.split('I could not produce a reflexion for this trial'))==1 else 'error'
                    self.collect_logs(self.reflexion, (True, reflexion_loglevel), (True, reflexion_loglevel), (False, reflexion_loglevel))
                    self.collect_logs(self.most_recent_reflexion, (False, reflexion_loglevel), (False, reflexion_loglevel), (True, reflexion_loglevel))

    async def _before_agent_episode_a(self, query: Optional[str]=None, reference: Optional[str]=None):
        self._before_agent_episode(query=query, reference=reference)


    async def _before_agent_episode_async(self, query: Optional[str]=None, reference: Optional[str]=None):
        await self._before_agent_episode_a(query=query, reference=reference)



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
            self.a, self.s_prime, self.is_termination_state  = self.agent_step(query)   
            if not self.is_termination_state:  
                self.intermediate_steps.append((self.a, self.s_prime))            
            
            self.done = True if (self.is_termination_state) or (self._is_halted(self.timestep)) else False
            if not self.done:
                self.timestep += 1


        # assessment the output
        if not self._is_halted(self.timestep):
            self.prediction = self.a.return_values['output'] 
        else: 
            self.prediction = "HALTED"
        self.judgement = self._assessment()
        self._add_judgement_to_agent_log()
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< run_agent_trials >>> '''
    def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Suppost Async by appending _async'''
        if self.is_reflexion_agent:
            self._reflexion_reset_()
            if reference==None:            
                raise ValueError("For Reflexion agent, reference should be provided for 'run_agent_trials' method.")
        self.trial=0
        self.judgement = ['', "PENDING"]
        
    async def _before_agent_trials_a(self, query: Optional[str]=None, reference: Optional[str]=None):
        self._before_agent_trials(query=query, reference=reference)
    
    async def _before_agent_trials_async(self, query: Optional[str]=None, reference: Optional[str]=None):
        await self._before_agent_trials_a(query=query, reference=reference)

    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None)-> None:
        self._before_agent_trials(query=query, reference=reference)
        self.collect_logs(f"----- New test point -----", (False, 'info'), (True, 'info'), (True, 'info'))
        self.collect_logs(f"Query: {query}", (True, 'info'), (True, 'info'), (False, 'info'))
        
        while self.judgement[1]!='CORRECT' and self.trial<num_trials:  
            self.collect_logs(f"Trial {self.trial+1}", (True, 'info'), (True, 'info'), (False, 'info'))
            self.run_agent_episode(query=query, reference=reference, trial=self.trial, single_episode=False)  
            self.trial += 1
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< Parsing into string >>>'''
    def _parsing_action_argument_value(self, value):
        try:
            float(value)
        except:
            value = "'"+value+"'"
        return value

    def _parsing_intermediate_steps_into_str(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        ''' Override this property for any child class IF NECESSARY'''
        raise NotImplementedError

    def _parsing_Thought_and_Action_into_str(self, data)-> Tuple[str, str]:
        ''' Support Async by appending _async '''
        Thought_loglevel = 'error'
        Action_loglevel = 'error'
        if self.thought_word!=None:
            thought, action = re.split(self.action_word, data)  
            if self.contains_word(action, "```json"):
                action = action.replace("```json", "```") # proactive correcting for gpt-4-turbo 1106
            try:
                Thought = self._parsing_thought_into_str(thought)
                Thought_loglevel = 'info'
            except Exception as e:
                Thought = f'{self.thought_word[:-1]} {self.timestep+1}: Failed to parse Thought into str. The original string is "{thought}"'
            self.collect_logs(Thought, (True, Thought_loglevel), (True, Thought_loglevel), (True, Thought_loglevel))        
        else:
            Thought = ''

        try:    
            action = data if self.thought_word==None else action
            Action = self._parsing_action_into_str(action)
            Action_loglevel = 'info'
        except Exception as e:
            Action = f'{self.action_word[:-1]} {self.timestep+1}: Failed to parse Action into str. The original string is "{action}"'
        self.collect_logs(Action, (True, Action_loglevel), (True, Action_loglevel), (True, Action_loglevel))
        return Thought, Action 

    async def _parsing_Thought_and_Action_into_str_a(self, data)-> Tuple[str, str]:   
        _, Action = self._parsing_Thought_and_Action_into_str(data) 
        return '', Action
    
    async def _parsing_Thought_and_Action_into_str_async(self, data)-> Tuple[str, str]:
        _, Action = await self._parsing_Thought_and_Action_into_str_a(data)        
        return '', Action 


    def _parsing_thought_into_str(self, raw_thought_string:str)-> str:
        try:    
            Thought = raw_thought_string.strip()
            Thought = f'{self.thought_word[:-1]} {self.timestep+1}: '+Thought  if len(Thought.split(self.thought_word))==1 else Thought.replace(f'{self.thought_word[:-1]}: ', f'{self.thought_word[:-1]} {self.timestep+1}: ')
            return Thought
        except Exception as e:
            raise Exception(e)

    def _parsing_action_into_str(self, raw_action_string:str)-> str:
        try:
            data = json.loads(raw_action_string.strip().strip(' `\n'))
            action = data.get('action', '')
            try:
                action_input = ', '.join(f'{k}={self._parsing_action_argument_value(v)}' for k, v in data.get('action_input', {}).items())  
            except AttributeError:
                action_input = data.get('action_input')

            Action = f'{self.action_word[:-1]}: {action}({action_input})'
            Action = Action.replace(f'{self.action_word[:-1]}: ', f'{self.action_word[:-1]} {self.timestep+1}: ')
            return Action
        except Exception as e:
            raise Exception(e)

    def _parse_into_func_name_args(self, name, **kwargs):
        args_list = ', '.join([f"{key}={value}" for key, value in kwargs.items()])  
        return f"{name}({args_list})"         
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< Reflection >>>'''
    def _do_reflexion_(self, trajectory_only_log_for_reflexion:str)-> str:
        ''' Support Async by appending _async '''
        raise NotImplementedError
    async def _do_reflexion_a(self, trajectory_only_log_for_reflexion:str)-> str:
        self._do_reflexion_(trajectory_only_log_for_reflexion)
    async def _do_reflexion_async(self, trajectory_only_log_for_reflexion:str)-> str:
        await self._do_reflexion_a(trajectory_only_log_for_reflexion)

    def _reflexion_reset_(self)-> None:
        ''' Support Async by appending _async '''
        raise NotImplementedError
    async def _reflexion_reset_a(self)-> None:
        ''' Support Async by appending _async '''
        self._reflexion_reset_()
    async def _reflexion_reset_async(self)-> None:
        ''' Support Async by appending _async '''
        await self._reflexion_reset_a()
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< QA assessment >>> '''
    def _assessment(self):
        ''' Suppoer Async by appending _async'''
        if self.prediction != 'HALTED':            
            evaluation = self.evaluator(
                query = self.query,
                prediction = self.prediction,
                reference = self.reference
            )['value'] if self.reference != '' else 'NO_REFERENCE'
        else:
            evaluation = 'HALTED'

        if evaluation == 'CORRECT':            
            judgement =  ['Jugdement: Your answer is correct.', 'CORRECT']
        elif evaluation == 'INCORRECT':
            judgement =  [f'Jugdement: Your answer is incorrect. The correct answer is "{self.reference}"', 'INCORRECT']
        elif evaluation == 'NO_REFERENCE':
            judgement =  [f'Jugdement: There is no reference.', 'NO_REFERENCE']
        else:
            judgement =  [f'Jugdement: You failed to provide an answer because you exceeded the permitted number of reasoning steps. You must give an answer within {self.horizon} steps.', 'HALTED']
        return judgement

    async def _assessment_a(self):
        return self._assessment()

    async def _assessment_async(self):
        result = await  self._assessment_a()
        return result
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ####################### Helper methods #######################
    def _is_halted(self, timestep:int)-> bool:
        return (self.timestep >= self.horizon-1) and (not self.is_termination_state)


    def _add_judgement_to_agent_log(self):
        if self.judgement[1] != 'INCORRECT':
            "Judgement does not contain the reference"
            self.agent_log += self.judgement[0]
            self.trajectory_only_log_for_reflexion = self.agent_log
            self.collect_logs(self.judgement[0], (True, 'info'), (True, 'info'), (True, 'info')) 
        else:
            self.trajectory_only_log_for_reflexion = copy.deepcopy(self.agent_log)
            self.agent_log += self.judgement[0]
            self.trajectory_only_log_for_reflexion += (self.judgement[0].split(" The correct answer is"))[0]
            self.collect_logs(self.judgement[0], (True, 'info'), (True, 'info'), (False, 'info')) 
            self.collect_logs((self.judgement[0].split(" The correct answer is"))[0], (False, 'info'), (False, 'info'), (True, 'info')) 
    
    async def _add_judgement_to_agent_log_a(self):
        self._add_judgement_to_agent_log()

    async def _add_judgement_to_agent_log_async(self):
        await self._add_judgement_to_agent_log_a()








            



























