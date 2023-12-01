from agents.parallel_func_calling_agent import OpenAIParallelFuntionCallingAgent
from typing import Any, List, Tuple, Optional, Dict
from utils.wrappers import retry
from openai import RateLimitError


class ReflexionOpenAIParallelFuntionCallingAgent(OpenAIParallelFuntionCallingAgent):
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


    @retry(allowed_exceptions=(RateLimitError,))
    def _invoke_agent_action(self, query):
        if not self.use_chat_completion_api:
            raise NotImplementedError
        else:         
            self.messages[1]= {'role':'user', 'content':self.base_human_prompt.format_messages(input=query, reflections=self.reflexion if self.trial>0 else '')[0].content}   
            return self.parsing_to_ai_msg_dict_parser(
                        self.azure_apenai_client.chat_completions_create(
                        messages=self.messages, 
                        tools=self.openai_functions
                        )
                    )


    ''' Reflect '''
    def _do_reflexion_(self, trajectory_only_log_for_reflexion:str)-> str:
        new_reflexion = self.reflexion_chain(trajectory_only_log_for_reflexion)
        self.reflexion += new_reflexion
        self.most_recent_reflexion = self.reflexion_header + new_reflexion

    ''' Reset reflextions '''
    def _reflexion_reset_(self)-> None:
        self.reflexion = ''+self.reflexion_header
        self.most_recent_reflexion = None