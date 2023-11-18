from agents.base_cumtom_agent import BaseCustomAgent
from langchain.tools.render import render_text_description_and_args
from typing import List, Tuple, Any, Dict, Optional, Literal
import json
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class ReActAgent(BaseCustomAgent):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def _before_agent_step(self):
        pass


    def _agent_reset(self, query: Optional[str]=None, reference: Optional[str]=None):
        if query != None:
            self.query = query
        if reference != None:
            self.reference = reference
        self.prediction = ''
        judgement = ''
        self.timestep: int = -1
        self.is_finished: bool = False
        self.result: Dict = None        
        self.intermediate_steps: List = []


    def _get_action_string(self, raw_action_string:str)-> str:
        data = json.loads(raw_action_string.strip().strip(' `\n'))  
        action = data.get('action', '')  
        try:
            action_input = ', '.join(f'{k}={v}' for k, v in data.get('action_input', {}).items())  
        except AttributeError:
            action_input = data.get('action_input')
        Action = f'Action: {action}({action_input})'
        Action = Action.replace('Action: ', f'Action: {self.timestep+1}: ')
        return Action


    def _format_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        return format_log_to_str(intermediate_steps)


    def _base_prompt_postprocessing(self)-> ChatPromptTemplate:
        return self.base_prompt.partial(
            tools=render_text_description_and_args(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
            )
    
    def _evaluation(self)-> str:
        if self.prediction != 'HALTED':
            evaluation = self.evaluator(
                query = self.query,
                prediction = self.prediction,
                reference = self.reference
            )['value']
        else:
            evaluation = 'HALTED'

        if evaluation == 'CORRECT':            
            judgement =  'Jugdement: Your answer is correct.'
        elif evaluation == 'INCORRECT':
            judgement =  f'Jugdement: Your answer is incorrect. The correct answer is "{self.reference}"'
        else:
            judgement =  f'Jugdement: You failed to provide an answer because you exceeded the permitted number of reasoning steps. You must give an answer within {self.max_trials} steps.'
        return judgement