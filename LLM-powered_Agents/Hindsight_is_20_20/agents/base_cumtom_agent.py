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





class BaseCustomAgent:

    def __init__(self,
                reasoninig_engine: Any,
                base_prompt: ChatPromptTemplate|str,
                schemas_and_tools: List[Tuple[BaseModel, Dict[str, Tool]]],
                evaluator: Any,
                action_word: str,
                thought_word: Optional[str]=None, 
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
        self.retry_RateLimitError = retry_RateLimitError
        self.retry_standby = retry_standby
        self.stop_words = stop_words
        self.horizon = horizon
        self.e2e_log_folder = e2e_log_folder
        self.trajectory_only_log_folder = trajectory_only_log_folder
        self.file_logging = file_logging
        self.colsole_logging = colsole_logging



        ### Induced attributes
        self.schemas = [schema for schema, tool in self.schemas_and_tools]
        self.tools = [tool for schema, tool in self.schemas_and_tools]
        self.tool_dictionary = {schema.__name__:tool for schema, tool in self.schemas_and_tools}
        self.e2e_logger = get_hd22_file_logger(log_file= os.path.join(self.e2e_log_folder, self.get_logger_name_prefix()+'_e2e.log'), logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_e2e' )
        self.trajectory_logger = get_hd22_file_logger(log_file= os.path.join(self.trajectory_only_log_folder, self.get_logger_name_prefix()+'_trajectory.log'), logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_trajectory' )
        self.console_logger = get_hd22_stream_logger(logger_name=self.get_logger_name_prefix()+str(datetime.now())+'_console')



        ### Intricsic parameters
        self.query: str = ''
        self.reference: str = ''
        self.done: bool = False
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





    ####################### Fundamental methods #######################
    def agent_reset(self):
        self._before_agent_episode()
        self._before_agent_trials(reference='Instance initialization')

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

    def get_logger_name_prefix(self):
        korea_time = datetime.now(pytz.timezone('Asia/Seoul'))  
        year, month, day = korea_time.year, korea_time.month, korea_time.day
        return f'{self.__class__.__name__}_{year}-{month}-{day}'

    def contains_word(self, sentence, word):  
        words = sentence.lower().split()    
        return word.lower() in words  

    def sleep(self, e):
        try:
            wait_time = int(str(e).split(' Please retry after ')[1].split(' seconds. ')[0])
        except:
            wait_time = self.retry_standby
        print(f'RateLimitError -----> Will automatically retry {wait_time} seconds later.')     
        for s in range(wait_time, 0, -1):
            print(s, end=' ')
            time.sleep(1)


    ####################### Simulation methods #######################
    def agent_step(self, query: str):
        """
        In Reinforcement-learning context, 'agent_step' method takes in 's' as input and returns 'a'.
        Here, the 'query', 'agent_action', and 'Observation' act as 's', 'a' and 's_prime', respectively.
        """
        self._before_agent_step()        
        # try: 
        #     FALG = True
        #     while FALG:
        #         try:
        #             agent_action = self._invoke_agent_action(query)
        #             FALG = False
        #         except RateLimitError as e:
        #             self.sleep(e)            
        #     Observation, temp_scratchpad = self._func_execution(agent_action=agent_action)
        
        try: 
            agent_action = self._invoke_agent_action(query)        
            Observation, temp_scratchpad = self._func_execution(agent_action=agent_action)

        except Exception as e:
            """ This catches the exception where the brain fails to produce AgentAction or AgentFinish.  """
            agent_action = self._invoke_agent_action_for_exception(e)
            Observation, temp_scratchpad = self._func_execution_for_exception(e)
        
        self.agent_log += temp_scratchpad
        done = True if isinstance(agent_action, AgentFinish) else False 
        return agent_action, Observation, done       


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
            if not done:  
                self.intermediate_steps.append((self.a, self.s_prime))            
            self.done = True if (done) or (self._is_halted(self.timestep)) else False

        # assessment the output
        if not self._is_halted(self.timestep):
            self.prediction = self.a.return_values['output'] 
        else: 
            self.prediction = "HALTED"
        self.judgement = self._assessment()
        self._add_judgement_to_agent_log()


    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None)-> None:
        self._before_agent_trials(query=query, reference=reference)
        self.collect_logs(f"----- New test point -----", (False, 'info'), (True, 'info'), (True, 'info'))
        self.collect_logs(f"Query: {query}", (True, 'info'), (True, 'info'), (False, 'info'))
        
        while self.judgement[1]!='CORRECT' and self.trial<num_trials:  
            self.collect_logs(f"Trial {self.trial+1}", (True, 'info'), (True, 'info'), (False, 'info'))
            self.run_agent_episode(query=query, reference=reference, trial=self.trial)  
            self.trial += 1



    ####################### Helper methods #######################
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

    def _assessment(self):
        ''' Override this property for any child class IF NECESSARY'''
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


    def _before_agent_step(self):
        ''' Override this property for any child class IF NECESSARY'''
        pass


    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Override this property for any child class IF NECESSARY'''
        self.query = query
        self.reference = reference if reference != None else ''
        self.agent_log = ''
        self.prediction = ''
        self.timestep = -1
        self.done  = False   
        self.intermediate_steps = []
        self.judgement = ['', "PENDING"]
        self.a = None
        self.s_prime = ''

    def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Override this property for any child class IF NECESSARY'''
        self.trial=0
        self.judgement = ['', "PENDING"]




    @retry(allowed_exceptions=(RateLimitError,))
    def _get_function_observation(self, tool, tool_input):
        return self.tool_dictionary[tool].run(**tool_input)

    # def _invoke_agent_action(self, query):
    #     ''' Override this property for any child class  IF NECESSARY'''
    #     return self.brain.invoke({
    #             'intermediate_steps': self.intermediate_steps,
    #             'input': query,
    #         })

    @retry(allowed_exceptions=(RateLimitError,))
    def _invoke_agent_action(self, query):
        ''' Override this property for any child class  IF NECESSARY'''
        return self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query,
            })

    def _invoke_agent_action_for_exception(self, e: Optional[str]=None):
        ''' Override this property for any child class  IF NECESSARY'''
        log = f'Exception raised. Neither AgentAction nor AgentFinish is produced. The error message is "{e}"' if e != None else 'Exception raised. Neither AgentAction nor AgentFinish is produced.'
        log += '\nAction:\n```\n{\n"action": "",\n"action_input": ""\n}\n```'
        return AgentAction(
                log=log,
                tool='',
                tool_input='',
                type = 'AgentAction')




    def _is_halted(self, timestep:int)-> bool:
        return self.timestep>self.horizon-2 and not isinstance(self.a, AgentFinish)


    def _parsing_action_argument_value(self, value):
        try:
            float(value)
        except:
            value = "'"+value+"'"
        return value


    def _parsing_intermediate_steps_into_str(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        ''' Override this property for any child class IF NECESSARY'''
        raise NotImplementedError



    def _parsing_Thought_and_Action_into_str(self, agent_action_log:str)-> Tuple[str, str]:
        Thought_loglevel = 'error'
        Action_loglevel = 'error'
        if self.thought_word!=None:
            thought, action = re.split(self.action_word, agent_action_log)  
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
            action = agent_action_log if self.thought_word==None else action
            Action = self._parsing_action_into_str(action)
            Action_loglevel = 'info'
        except Exception as e:
            Action = f'{self.action_word[:-1]} {self.timestep+1}: Failed to parse Action into str. The original string is "{action}"'
        self.collect_logs(Action, (True, Action_loglevel), (True, Action_loglevel), (True, Action_loglevel))

        return Thought, Action 


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






