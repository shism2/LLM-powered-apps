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

class BaseCustomAgent:
    def __init__(self,
                base_llm: Any,
                base_prompt: ChatPromptTemplate|str,
                tools: List[Tool],
                evaluator: Any,
                thought_word: str, 
                action_word: str,
                stop_words: List[str]|None=None,
                horizon: int=6,
                print_stdout: bool=True):
        
        # reasoning engine
        self.base_llm: Any = base_llm
        
        # tools
        self.tools = tools
        self.tool_dictionary = { tool.name:tool for tool in self.tools}

        # evaluator
        self.evaluator = evaluator

        # agent attributes
        self.query = ''
        self.reference = ''
        self.prediction = ''
        self.judgement = ['', 0]
        self.thought_word = thought_word
        self.action_word = action_word
        self.horizon = horizon
        self.print_stdout = print_stdout
        self.stop_words = stop_words
        self.agent_log: List = ['']

        # prompt
        if isinstance(base_prompt, str):
            base_prompt = hub.pull(base_prompt)
        if len(base_prompt)!=2 or not isinstance(base_prompt[0], SystemMessagePromptTemplate) or not isinstance(base_prompt[1], HumanMessagePromptTemplate): 
            raise ValueError("Error in 'base_prompt'.")
        self.base_system_prompt = base_prompt[0]
        self.base_human_prompt = base_prompt[1]


        # agent_reset
        self._agent_reset()


    @property
    def base_prompt(self)-> ChatPromptTemplate:
        '''
        Override this property for any child class
        '''
        raise NotImplementedError

    @property
    def brain(self)-> Any:
        '''
        Override this property for any child class
        '''
        raise NotImplementedError


    def print_on_stdout(self, string: str, sep='\n')->None:
        if self.print_stdout:
            print(string, sep=sep)


    def agent_log_reset(self)-> None:
        self.agent_log = ['']


    def agent_run(self, query: str, reference: Optional[str]=None, multiple_trials: bool=False, agent_log_reset: bool=False)-> None:    

        self._agent_reset(query=query, reference=reference)
        if agent_log_reset:
            self.agent_log_reset()
        if self.agent_log[-1] != '':
            self.agent_log.append('')    
        self.agent_log[-1] += f"Qurey: {query}\n"
        if self.print_stdout and multiple_trials==False:
            print(f"Query: {query}")
        
        while (not self.is_finished) and (not self.is_halted(self.timestep)):
            self.timestep += 1

            self.is_finished, self.result = self.agent_step(query)

        if reference != None:
            self.prediction = self.result.return_values['output'] if isinstance(self.result, AgentFinish) else "HALTED"
            self.judgement = self._evaluation()
            self.execution(agent_action='NO_NEED', judgement=True)


    def agent_run_miltiple_trials(self, num_trials: int, query: str, reference: Optional[str]=None, agent_log_reset=True)-> None:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError



    def agent_step(self, query: str)-> Tuple[bool, AgentFinish|None]:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError
           


    def execution(self, agent_action: AgentAction|AgentFinish|Literal['NO_NEED'], judgement=False)-> str|None:
        if judgement:
            self.agent_log[-1] += self.judgement[0]
            self.print_on_stdout(self.judgement[0])
            return


        Thought, Action = self._get_Thought_and_Action(agent_action.log)
        
        # Observation

        try:
            if isinstance(agent_action, AgentAction):
                observation = self.tool_dictionary[agent_action.tool].run(agent_action.tool_input)
                Observation = (f'Observation {self.timestep+1}: '+observation).rstrip('\n')
                self.print_on_stdout(Observation, sep='')
                return observation, Thought+'\n'+Action+'\n'+Observation+'\n'
            else:
                Observation = (f'Answer: '+agent_action.return_values['output']).rstrip('\n') 
                self.print_on_stdout(Observation, sep='')
                return None, Thought+'\n'+Action+'\n'+Observation+'\n'
        except Exception as e:
            return None, 'Exception'




    def _get_Thought_and_Action(self, agent_action_log:str, print_on_stdout=True)-> Tuple[str, str]:
        thought, action = re.split(self.action_word, agent_action_log)  
        
        # Thought
        Thought = thought.strip()
        Thought = f'Thought {self.timestep+1}: '+Thought if len(Thought.split(self.thought_word))==1 else Thought.replace('Thought: ', f'Thought {self.timestep+1}: ')
        if print_on_stdout:
            self.print_on_stdout(Thought, sep='')
        
        # Action
        Action = self._get_action_string(action)
        if print_on_stdout:
            self.print_on_stdout(Action, sep='')
        return Thought, Action



    def is_halted(self, timestep:int)-> bool:
        return self.timestep>self.horizon-2 and self.result==None


    def change_prompt(self, new_prompt: ChatPromptTemplate)-> None:
        self.base_prompt = new_prompt


    def _before_agent_step(self):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _agent_reset(self, query: Optional[str]=None, reference: Optional[str]=None):
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





