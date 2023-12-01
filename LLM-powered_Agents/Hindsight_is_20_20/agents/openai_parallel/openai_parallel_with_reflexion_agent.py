from agents.openai_parallel.openai_parallel_agent import OpenAIParallelAgent
from typing import Any, List, Tuple, Optional, Dict
from utils.wrappers import retry
from openai import RateLimitError


class OpenAIParallelWithReflexionAgent(OpenAIParallelAgent):
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

    ''' <<< Invoke Brain >>> '''
    async def _populate_user_message_async(self, query):
        self.messages[1]= {'role':'user', 'content':self.base_human_prompt.format_messages(input=query, reflections=self.reflexion if self.trial>0 else '')[0].content}  
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' <<< Reflection >>>'''
    def do_reflexion(self, trajectory_only_log_for_reflexion:str)-> str:
        ''' Support Async by appending _async '''
        new_reflexion = self.reflexion_chain(trajectory_only_log_for_reflexion)
        self.reflexion += new_reflexion
        self.most_recent_reflexion = self.reflexion_header + new_reflexion

    def reflexion_reset(self)-> None:
        ''' Support Async by appending _async '''
        self.reflexion = ''+self.reflexion_header
        self.most_recent_reflexion = None
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



