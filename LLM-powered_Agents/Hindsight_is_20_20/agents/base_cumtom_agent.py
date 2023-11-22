from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Tuple, Any, Dict, Optional, Literal
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import Tool
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain import hub
from langchain.tools.render import render_text_description_and_args
import copy
import json  
import re  
from typing import Optional
import logging
from loggers.get_loggers import get_hd22_file_logger, get_hd22_stream_logger
import os
import datetime

class BaseCustomAgent:
    def __init__(self,
                reasoninig_engine: Any,
                base_prompt: ChatPromptTemplate|str,
                tools: List[Tool],
                evaluator: Any,
                thought_word: Optional[str]=None, 
                action_word: Optional[str]=None,
                stop_words: List[str]|None=None,
                horizon: int=6,
                e2e_log_folder:str='loggers/e2e_logs',
                trajectory_log_folder:str='loggers/trajectory_logs',
                file_logging:bool=False,
                colsole_logging: bool=True,
                ):
        
        # reasoning engine
        self.reasoninig_engine: Any = reasoninig_engine
        
        # tools
        self.tools = tools
        self.tool_dictionary = { tool.name:tool for tool in self.tools}

        # evaluator
        self.evaluator = evaluator

        # agent attributes
        self.agent_observation: Dict = None
        self.query = ''
        self.reference = ''
        self.prediction = ''
        self.judgement: List = ['', 0]
        self.thought_word = thought_word
        self.action_word = action_word
        self.horizon = horizon
        self.colsole_logging = colsole_logging
        self.stop_words = stop_words
        self.agent_log: str = ''
        self.agent_log_for_trajectory: str = ''
        self.e2e_log_folder = e2e_log_folder
        self.trajectory_log_folder = trajectory_log_folder
        self.file_logging = file_logging

        # prompt
        if isinstance(base_prompt, str):
            base_prompt = hub.pull(base_prompt)
        if len(base_prompt)!=2 or not isinstance(base_prompt[0], SystemMessagePromptTemplate) or not isinstance(base_prompt[1], HumanMessagePromptTemplate): 
            raise ValueError("Error in 'base_prompt'.")
        self.base_system_prompt = base_prompt[0]
        self.base_human_prompt = base_prompt[1]


        # loggers
        self.e2e_logger = get_hd22_file_logger(log_file= os.path.join(self.e2e_log_folder, self.get_log_prefix()+'_e2e.log'), logger_name=self.get_log_prefix()+str(datetime.datetime.now())+'_e2e' )
        self.trajectory_logger = get_hd22_file_logger(log_file= os.path.join(self.trajectory_log_folder, self.get_log_prefix()+'_trajectory.log'), logger_name=self.get_log_prefix()+str(datetime.datetime.now())+'_trajectory' )
        self.console_logger = get_hd22_stream_logger(logger_name=self.get_log_prefix()+str(datetime.datetime.now())+'_console')

        # agent_reset
        self.agent_reset()

    #############################
    #### Fundamental methods ####
    #############################
    @property
    def base_prompt(self)-> ChatPromptTemplate:
        '''
        Override this property for any child class
        '''
        raise NotImplementedError

    @base_prompt.setter
    def base_prompt(self, new_prompt: ChatPromptTemplate)-> None:
        '''
        Override this property setter for any child class
        '''
        raise NotImplementedError    

    @property
    def brain(self)-> Any:
        '''
        Override this property for any child class
        '''
        raise NotImplementedError


    def agent_reset(self):
        self._before_agent_episode(query='')



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
      

    def clean_logs(self):
        with open(os.path.join(self.e2e_log_folder, self.get_log_prefix()+'_e2e.log'), 'w') as f:
            f.write('')
        with open(os.path.join(self.trajectory_log_folder, self.get_log_prefix()+'_trajectory.log'), 'w') as f:
            f.write('')


    def get_log_prefix(self):
        '''
        Override this property for any child class
        '''
        raise NotImplementedError




    ####################################
    #### Agentic simulation methods ####
    ####################################
    def agent_step(self, query: str)-> Tuple[bool, AgentFinish|None]:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError

    def func_execution(self, agent_action: AgentAction|AgentFinish|Literal['NO_NEED'])-> Tuple[str,str]:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def run_agent_episode(self, query: str, reference: Optional[str]=None, trial: int=0, single_trial=False)-> None:    
        self._before_agent_episode(query=query, reference=reference)

        self.agent_log += f"Qurey: {query}\n"
        if single_trial:
            self.collect_logs(f"Trial {trial+1}", (True, 'info'), (True, 'info'), (True, 'info'))
            self.collect_logs(f"Query: {query}", (True, 'info'), (True, 'info'), (True, 'info'))
        else:
            self.collect_logs(f"Trial {trial+1}", (False, 'info'), (False, 'info'), (True, 'info'))
            self.collect_logs(f"Query: {query}", (False, 'info'), (False, 'info'), (True, 'info'))
        
        while (not self.is_finished) and (not self._is_halted(self.timestep)):
            self.timestep += 1
            self.is_finished, self.agent_observation = self.agent_step(query)

        if reference != None:
            '''
            At this stage, the agents either gave an answer or has been halted.
            '''
            self.prediction = self.agent_observation.return_values['output'] if isinstance(self.agent_observation, AgentFinish) else "HALTED"
            self.judgement = self._evaluation()
            self._add_judgement_to_agent_log()


    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None)-> None:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError



    ###########################################
    #### Agentic-simulation helper methods ####
    ###########################################
    def _get_Thought_and_Action(self, agent_action_log:str)-> Tuple[str, str]:
        Thought_loglevel = 'error'
        Action_loglevel = 'error'
        try:
            thought, action = re.split(self.action_word, agent_action_log)  
            try:    
                Thought = thought.strip()
                Thought = f'Thought {self.timestep+1}: '+Thought if len(Thought.split(self.thought_word))==1 else Thought.replace('Thought: ', f'Thought {self.timestep+1}: ')
                Thought_loglevel = 'info'
            except Exception as e:
                Thought = f'Thought {self.timestep+1}: Failed to parse Thought into str. The error message is "{e}"'
    
            try:    
                Action = self._get_action_string(action)
                Action_loglevel = 'info'
            except Exception as e:
                Action = f'Action {self.timestep+1}: Failed to parse Action into str. The error message is "{e}"'
        except Exception as e:
            Thought = f'Thought {self.timestep+1}: Failed to parse Thought into str. The error message is "{e}"'
            Action = f'Action {self.timestep+1}: Failed to parse Action into str. The error message is "{e}"'
        finally:
            self.collect_logs(Thought, (True, Thought_loglevel), (True, Thought_loglevel), (True, Thought_loglevel))
            self.collect_logs(Action, (True, Action_loglevel), (True, Action_loglevel), (True, Action_loglevel))
            return Thought, Action

    def _add_judgement_to_agent_log(self):
        if self.judgement[1] != 'INCORRECT':
            "Judgement does not contain the reference"
            self.agent_log += self.judgement[0]
            self.agent_log_for_trajectory = self.agent_log
            self.collect_logs(self.judgement[0], (True, 'info'), (True, 'info'), (True, 'info')) 
        else:
            self.agent_log_for_trajectory = copy.deepcopy(self.agent_log)
            self.agent_log += self.judgement[0]
            self.agent_log_for_trajectory += (self.judgement[0].split(" The correct answer is"))[0]
            self.collect_logs(self.judgement[0], (True, 'info'), (True, 'info'), (False, 'info')) 
            self.collect_logs((self.judgement[0].split(" The correct answer is"))[0], (False, 'info'), (False, 'info'), (True, 'info')) 

  

    def _is_halted(self, timestep:int)-> bool:
        return self.timestep>self.horizon-2 and self.agent_observation==None


    def _before_agent_step(self):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _before_agent_trials(self):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _get_action_string(self, raw_action_string:str)-> str:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _format_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _base_prompt_postprocessing(self)-> ChatPromptTemplate:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _evaluation(self):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError





