from agents.base_cumtom_agent import BaseCustomAgent
from agents.react_agents import ReActAgent
from typing import Any, List, Tuple, Optional, Dict
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from typing import Literal
from datetime import datetime
import pytz

class ReflexionReActAgent(ReActAgent):
    def __init__(self, 
                reflexion_chain: Any,
                reflexion_header: str, 
                **kwargs):

        self.reflexion_chain = reflexion_chain
        self.reflexion_header = reflexion_header
        self.reflexion = ''+self.reflexion_header
        self.most_recent_reflexion = None
        
        super().__init__(**kwargs)





 
    ''' Pre-process for running a new episode : OVERRIDE '''
    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        ''' Override this property for any child class IF NECESSARY'''
        self.query = query
        self.reference = reference if reference != None else ''
        self.agent_log = ''
        self.prediction = ''
        self.timestep = -1
        self.done = False    
        self.intermediate_steps = []
        self.judgement = ['', 0]
        self.a = None
        self.s_prime = ''
        if self.trial>0:
                self.collect_logs(f"Reflexion......", (True, 'info'), (True, 'info'), (False, 'info'))            
                self.do_reflexion(self.trajectory_only_log_for_reflexion)
                reflexion_loglevel = 'info' if len(self.most_recent_reflexion.split('I could not produce a reflexion for this trial'))==1 else 'error'
                self.collect_logs(self.reflexion, (True, reflexion_loglevel), (True, reflexion_loglevel), (False, reflexion_loglevel))
                self.collect_logs(self.most_recent_reflexion, (False, reflexion_loglevel), (False, reflexion_loglevel), (True, reflexion_loglevel))



    ''' Pre-process for running new trials : OVERRIDE '''
    def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
        if reference==None:            
            raise ValueError("For Reflexion agent, reference should be provided for 'run_agent_trials' method.")
        self.trial=0
        self.judgement = ['', 0]
        self.reflexion_reset()


    ''' Trigger Brain (agent chain) : OVERRIDE '''
    def _invoke_agent_action(self, query):
        return self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query,
                'reflections':self.reflexion, 
            })


    ''' Reflect '''
    def do_reflexion(self, trajectory_only_log_for_reflexion:str)-> str:

        new_reflexion = self.reflexion_chain(trajectory_only_log_for_reflexion)
        self.reflexion += new_reflexion
        self.most_recent_reflexion = self.reflexion_header + new_reflexion

    ''' Reset reflextions '''
    def reflexion_reset(self)-> None:
        self.reflexion = ''+self.reflexion_header
        self.most_recent_reflexion = None

 







