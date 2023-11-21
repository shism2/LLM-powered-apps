from agents.base_cumtom_agent import BaseCustomAgent
from langchain.tools.render import render_text_description_and_args
from typing import List, Tuple, Any, Dict, Optional, Literal
import json
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
import re
from langchain.schema.runnable import RunnableLambda

class ReActAgent(BaseCustomAgent):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def _before_agent_step(self):
        pass


    def _before_agent_episode(self, query: Optional[str]=None, reference: Optional[str]=None):
        """Before staring a new episode"""
        if query != None:
            self.query = query
        if reference != None:
            self.reference = reference
        self.prediction = ''
        self.timestep: int = -1
        self.is_finished: bool = False
        self.result: Dict = None        
        self.intermediate_steps: List = []
        self.judgement = ['', 0]



    def agent_step(self, query: str)-> Tuple[bool, AgentFinish|None]:
        '''
        Override this method for any child class
        '''
        self._before_agent_step()
        try:
            agent_action = self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query,
            })
        except Exception as e:
            """ This catches the exception where the brain fails to produce AgentAction or AgentFinish.  """
            agent_action = AgentAction(
                log='Thought: Unexpected exception has been raised. Brain cannot produce AgentAction or AgentFinish. ' + f'The error message is "{e}".'+ '\nAction:\n```\n{\n"action": "",\n"action_input": ""\n}\n```',
                tool='',
                tool_input='',
                type = 'AgentAction')
        finally:
            observation, agent_log = self.execution(agent_action=agent_action)
            self.agent_log[-1] += agent_log

            if isinstance(agent_action, AgentFinish):            
                return True, agent_action        
            else:    
                self.intermediate_steps.append((agent_action, observation))
                return False, None            


    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None, agent_log_reset=True)-> None:
        '''
        Override this method for any child class
        '''
        self.print_on_stdout(f"Query: {query}")

        if agent_log_reset:
            self.agent_log_reset()
        
        trial=0
        while self.judgement[1]!=1 and trial<num_trials:
            self.print_on_stdout(f"---- Trial {trial+1} ----")
            self.run_agent_episode(query=query, reference=reference, multiple_trials=True)  
            trial += 1


    def _get_action_string(self, raw_action_string:str)-> str:
        try:
            data = json.loads(raw_action_string.strip().strip(' `\n'))  
            action = data.get('action', '')  
            try:
                action_input = ', '.join(f'{k}={v}' for k, v in data.get('action_input', {}).items())  
            except AttributeError:
                action_input = data.get('action_input')
            Action = f'Action: {action}({action_input})'
            Action = Action.replace('Action: ', f'Action {self.timestep+1}: ')
        except Exception as e:
            Action = f'Action {self.timestep+1}: Filed to parse Action. The error message is "{e}"'
        finally:
            return Action


    def _format_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        return format_log_to_str(intermediate_steps)



    @property
    def base_prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([self.base_system_prompt, self.base_human_prompt])
        prompt = prompt.partial(
            tools=render_text_description_and_args(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
            )
        return prompt

    @property
    def brain(self)-> Any:

        def fix_json(ai_message_chunk):  
            '''
            This function is need for gpt-4-turbo for avoding error when invoking 'parse_json_markdown()'
            '''
            input_str = ai_message_chunk.content 
            start = input_str.find('"action": ') + len('"action": ')  
            end = input_str.find(',', start)  
            
            action_value = input_str[start:end].strip()  
            
            # Check if action_value is already enclosed in double quotes  
            if not (action_value.startswith('"') and action_value.endswith('"')):  
                fixed_str = input_str[:start] + '"' + action_value + '"' + input_str[end:]  
            else:  
                fixed_str = input_str    

            ai_message_chunk.content = fixed_str        
            return ai_message_chunk 


        async def fix_json_async(ai_message_chunk):  
            '''
            This function is need for gpt-4-turbo for avoding error when invoking 'parse_json_markdown()'
            '''
            input_str = ai_message_chunk.content 
            start = input_str.find('"action": ') + len('"action": ')  
            end = input_str.find(',', start)  
            
            action_value = input_str[start:end].strip()  
            
            # Check if action_value is already enclosed in double quotes  
            if not (action_value.startswith('"') and action_value.endswith('"')):  
                fixed_str = input_str[:start] + '"' + action_value + '"' + input_str[end:]  
            else:  
                fixed_str = input_str    

            ai_message_chunk.content = fixed_str        
            return ai_message_chunk 


        brain = (
            RunnablePassthrough.assign(agent_scratchpad  = lambda x: self._format_scratchpad(x["intermediate_steps"]),) 
            | self.base_prompt
            | self.base_llm.bind(stop=self.stop_words)
            | RunnableLambda(fix_json, afunc=fix_json_async)
            | JSONAgentOutputParser()
        )  
        return brain      


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
            judgement =  ['Jugdement: Your answer is correct.', 1]
        elif evaluation == 'INCORRECT':
            judgement =  [f'Jugdement: Your answer is incorrect. The correct answer is "{self.reference}"', 0]
        else:
            judgement =  [f'Jugdement: You failed to provide an answer because you exceeded the permitted number of reasoning steps. You must give an answer within {self.horizon} steps.', -1]
        return judgement