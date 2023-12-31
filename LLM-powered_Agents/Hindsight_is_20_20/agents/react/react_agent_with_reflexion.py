from agents.react.react_agent import ReActAgent
from typing import Any, List, Tuple, Optional, Dict
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from typing import Literal
from datetime import datetime
import pytz
from utils.wrappers import retry
from openai import RateLimitError


class ReActWithReflexionAgent(ReActAgent):
    @property
    def is_reflexion_agent(self):
        return True

    def __init__(self, 
                reflexion_chain: Any,
                reflexion_header: str, 
                **kwargs):

        self.reflexion_chain = reflexion_chain
        self.reflexion_header = reflexion_header
        self.reflexion = ''+self.reflexion_header
        self.most_recent_reflexion = None        
        super().__init__(**kwargs)

    ''' Trigger Brain (agent chain) : OVERRIDE '''
    @retry(allowed_exceptions=(RateLimitError,))
    def invoke_agent_action(self, query):
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




 







