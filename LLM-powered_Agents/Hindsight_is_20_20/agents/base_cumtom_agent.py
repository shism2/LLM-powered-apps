from utils.agent_components.get_llm import LangChainLLMWrapper
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Tuple, Any, Dict, Optional, Literal
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import Tool
from langchain.agents.format_scratchpad.log import format_log_to_str
import copy
import json  
import re  

class BaseCustomAgent:
    def __init__(self,
                base_llm:LangChainLLMWrapper,
                initial_prompt: ChatPromptTemplate,
                tools: List[Tool],
                thought_word: str, 
                action_word: str,
                stop_words: List[str]|None=None,
                max_trials: int=6,
                print_stdout: bool=True):
        
        # reasoning engine
        self.base_llm: Any = base_llm.llm
        
        # tools
        self.tools = tools
        self.tool_dictionary = { tool.name:tool for tool in self.tools}

        # agent attributes
        self.thought_word = thought_word
        self.action_word = action_word
        self.max_trials = max_trials
        self.print_stdout = print_stdout
        self.stop_words = stop_words

        # prompt to brain
        if len(initial_prompt)!=2 or not isinstance(initial_prompt[0], SystemMessagePromptTemplate) or not isinstance(initial_prompt[1], HumanMessagePromptTemplate): 
            raise ValueError("Error in 'initial_prompt'.")
        self.initial_prompt: ChatPromptTemplate = initial_prompt        
        self.system_prompt = copy.deepcopy(self.initial_prompt[0])
        self.human_prompt = copy.deepcopy(self.initial_prompt[1])        
        self.prompt = ChatPromptTemplate.from_messages([self.system_prompt, self.human_prompt])
        self.prompt = self._prompt_postprocessing()

        # Brain    
        self.brain = (
            RunnablePassthrough.assign(agent_scratchpad  = lambda x: self._format_scratchpad(x["intermediate_steps"]),) 
            | self.prompt
            | self.base_llm.bind(stop=self.stop_words)
            | JSONAgentOutputParser()
        )

        # agent_reset
        self._agent_reset()


    ### generic method ###
    def agent_run(self, query: str, episode: int=1)-> None:    
        self._agent_reset()
        self.agent_log.append(query)

        if self.print_stdout:
            print(f"-- Episode {episode} started. --")
            print(f"Query: {query}")
        
        while (not self.is_finished) and (not self.is_halted(self.timestep)):
            self.timestep += 1

            intermediate_steps_size, agent_log_size = len(self.intermediate_steps), len(self.agent_log)
            try:
                self.is_finished, self.result = self.agent_step(query, self.intermediate_steps)
            except:
                if len(self.intermediate_steps)>intermediate_steps_size:
                    self.intermediate_steps.pop()
                if len(self.agent_log)>agent_log_size:
                    self.agent_log.pop()
                self.timestep -= 1

        
        if self.is_halted(self.timestep):
            self._get_log(intermediate_step=None, is_halted=True)
    

    ### generic method ###
    def agent_step(self, query: str, intermediate_steps: List[Tuple[AgentAction, str]]=[])-> Tuple[bool, AgentFinish|None]:
        self._before_agent_step()

        agent_action = self.brain.invoke({
            'intermediate_steps': self.intermediate_steps,
            'input': query,
        })
           
            
        if isinstance(agent_action, AgentFinish):
            self._get_log(intermediate_step=agent_action, is_halted=False)
            return True, agent_action        
        else:    
            observation = self.tool_dictionary[agent_action.tool].run(agent_action.tool_input)
            self._get_log(intermediate_step=(agent_action, observation), is_halted=False)
            self.intermediate_steps.append((agent_action, observation))
            return False, None

            


    ### generic method ###
    def _get_log(self, intermediate_step: Tuple[AgentAction, str]|AgentFinish|None, is_halted=False)->None:
        if not is_halted:
            agent_log = intermediate_step[0].log if isinstance(intermediate_step, tuple) else intermediate_step.log
            
            Observation = (f'Observation {self.timestep+1}: '+intermediate_step[1]).rstrip('\n') if isinstance(intermediate_step, tuple) else (f'Answer: '+intermediate_step.return_values['output']).rstrip('\n') 
            
            thought, action = re.split(self.action_word, agent_log)  
            Thought = thought.strip()+'\n'
            Thought = f'Thought {self.timestep+1}: '+Thought if len(Thought.split(self.thought_word))==1 else Thought.replace('Thought: ', f'Thought {self.timestep+1}: ')
            
            Action = self._get_action_string(action)

            self.agent_log.append(Thought+Action+Observation)
        else:
            self.agent_log.append(f"Error: The length of episode reached the limit {self.max_trials} before getting the answer.")
        
        if self.print_stdout:
            print(self.agent_log[-1])


    ### generic method ###
    def is_halted(self, timestep:int)-> bool:
        return self.timestep>self.max_trials-2 and self.result==None


    ### generic method ###
    def change_prompt(self, new_prompt: ChatPromptTemplate)-> None:
        self.prompt = new_prompt


    def _before_agent_step(self):
        '''
        Override this method for any child class
        '''
        raise NotImplementedError


    def _agent_reset(self):
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


    def _prompt_postprocessing(self)-> ChatPromptTemplate:
        '''
        Override this method for any child class
        '''
        raise NotImplementedError






