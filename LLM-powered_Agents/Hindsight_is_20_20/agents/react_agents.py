from agents.base_cumtom_agent import BaseCustomAgent
from langchain.tools.render import render_text_description_and_args
from typing import List, Tuple, Any, Dict, Optional, Literal, Type
from pydantic import BaseModel, Field
from langchain.agents import Tool
import json
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
import re
from langchain.schema.runnable import RunnableLambda
from datetime import datetime
import pytz

class ReActAgent(BaseCustomAgent):
    @property
    def is_reflexion_agent(self):
        return False

    @property
    def base_prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([self.base_system_prompt, self.base_human_prompt])
        prompt = prompt.partial(
            tools=self._tool_description_for_system_msg_(self.schemas, self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
            )
        return prompt


    @property
    def brain(self)-> Any:

        def fix_json(ai_message_chunk):  
            # This function is need for gpt-4-turbo for avoding error when invoking 'parse_json_markdown()'
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
            # This function is need for gpt-4-turbo for avoding error when invoking 'parse_json_markdown()'
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
            RunnablePassthrough.assign(agent_scratchpad  = lambda x: self._parsing_intermediate_steps_into_str(x["intermediate_steps"]),) 
            | self.base_prompt
            | self.reasoninig_engine.bind(stop=self.stop_words)
            | RunnableLambda(fix_json, afunc=fix_json_async)
            | JSONAgentOutputParser()
        )  
        return brain     


    def __init__(self, **kwargs):
            super().__init__(**kwargs)


    def _tool_description_for_system_msg_(self, schemas: List[Type[BaseModel]], tools: List[Tool])->str:
        """adapted from langchain's render_text_description_and_args """
        tool_strings = []
        for schema, tool in zip(schemas, tools):
            tool_strings.append(f"{tool.name}: {tool.description}, args: {str(schema.schema()['properties'])}")
        return "\n".join(tool_strings)

 


    ''' Define parsing function of intermediate steps : OVERRIDE'''
    def _parsing_intermediate_steps_into_str(self, intermediate_steps: List[Tuple[AgentAction, str]])-> None:
        return format_log_to_str(intermediate_steps)



