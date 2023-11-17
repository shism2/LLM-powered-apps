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


    def _agent_reset(self):
        self.timestep: int = -1
        self.is_finished: bool = False
        self.result: Dict = None
        self.agent_log: List = []
        self.intermediate_steps: List = []


    def _get_action_string(self, raw_action_string:str)-> str:
        data = json.loads(raw_action_string.strip().strip(' `\n'))  
        action = data.get('action', '')  
        try:
            action_input = ', '.join(f'{k}={v}' for k, v in data.get('action_input', {}).items())  
        except AttributeError:
            action_input = data.get('action_input')
        Action = f'Action: {action}({action_input})'+'\n'  
        Action = Action.replace('Action: ', f'Action: {self.timestep+1}: ')
        return Action


    def _format_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        return format_log_to_str(intermediate_steps)


    def _prompt_postprocessing(self)-> ChatPromptTemplate:
        return self.prompt.partial(
            tools=render_text_description_and_args(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
            )
    
