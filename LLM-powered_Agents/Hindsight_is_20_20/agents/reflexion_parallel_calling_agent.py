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


    # def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
    #     ''' Override this property for any child class IF NECESSARY'''
    #     self.query = query
    #     self.reference = reference if reference != None else ''
    #     self.agent_log = ''
    #     self.prediction = ''
    #     self.timestep = 0
    #     self.done = False    
    #     self.is_termination_state = False
    #     self.intermediate_steps = []
    #     self.judgement = ['', 0]
    #     self.a = None
    #     self.s_prime = ''
    #     if self.trial>0:
    #         self.collect_logs(f"Reflexion......", (True, 'info'), (True, 'info'), (False, 'info'))            
    #         self._do_reflexion_(self.trajectory_only_log_for_reflexion)
    #         reflexion_loglevel = 'info' if len(self.most_recent_reflexion.split('I could not produce a reflexion for this trial'))==1 else 'error'
    #         self.collect_logs(self.reflexion, (True, reflexion_loglevel), (True, reflexion_loglevel), (False, reflexion_loglevel))
    #         self.collect_logs(self.most_recent_reflexion, (False, reflexion_loglevel), (False, reflexion_loglevel), (True, reflexion_loglevel))
    #     if self.use_chat_completion_api:
    #         self.messages = []
    #         self.messages.append({'role':'system', 'content':self.base_system_prompt.prompt.template})
    #         self.messages.append({'role':'user', 'content':''})



    # def _before_agent_trials(self, query: Optional[str]=None, reference: Optional[str]=None):
    #     if reference==None:            
    #         raise ValueError("For Reflexion agent, reference should be provided for 'run_agent_trials' method.")
    #     self.trial=0
    #     self.judgement = ['', "PENDING"]
    #     self._reflexion_reset_()
    #     if self.use_chat_completion_api:
    #         self.messages = []
    #         self.messages.append({'role':'system', 'content':self.base_system_prompt.prompt.template})
    #         self.messages.append({'role':'user', 'content':''})


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