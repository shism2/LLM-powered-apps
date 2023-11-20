from agents.base_cumtom_agent import BaseCustomAgent
from agents.react_agents import ReActAgent
from typing import Any, List, Tuple, Optional
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish

class ReflexionReActAgent(ReActAgent):
    def __init__(self, 
                reflexion_chain: Any,
                reflexion_header: str, 
                **kwargs):
        super().__init__(**kwargs)

        self.reflexion_chain = reflexion_chain
        self.reflexion_header = reflexion_header

        self.reflexion = ''+self.reflexion_header


    def do_reflexion(self, agent_log:str)-> str:
        reflexion = self.reflexion_chain(agent_log)
        self.reflexion += reflexion+'\n'

    def reflexion_reset(self)-> None:
        self.reflexion = ''+self.reflexion_header

 


    def agent_run_miltiple_trials(self, num_trials: int, query: str, reference: Optional[str]=None, agent_log_reset=True)-> None:
        '''
        Override this method for any child class
        '''
        if reference==None:
            raise ValueError("For Reflexion agent, reference should be provided for 'agent_run_miltiple_trials' method.")
        self.print_on_stdout(f"Query: {query}")

        if agent_log_reset:
            self.agent_log_reset()
            self.reflexion_reset()
        
        for trial in range(num_trials):
            if self.judgement[1] == 1:
                break
            if self.print_stdout:
                print(f"---- Trial {trial+1} ----")
                if trial>0:
                        print(f"Reflexion......")
                        self.do_reflexion(self.agent_log[-1])
                        print(f"{self.reflexion}")

            self.agent_run(query=query, reference=reference, multiple_trials=True) 




    def agent_step(self, query: str)-> Tuple[bool, AgentFinish|None]:
        '''
        Override this method for any child class
        '''
        self._before_agent_step()
        try:
            agent_action = self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query,
                'reflections':self.reflexion, 
            })
            observation, agent_log = self.execution(agent_action=agent_action)
            if observation == 'Exception':
                Thought, Action = self._get_Thought_and_Action(agent_action.log, print_on_stdout=False)
                observation = f'Observation {self.timestep+1}: Unexpected Exception has been raised. Couldn\'t get observation(function output.) The error message is "{agent_log}". I should try again.'
                self.print_on_stdout(observation)
                agent_log = Thought+'\n'+Action+'\n'+ observation
            self.agent_log[-1] += agent_log


            if isinstance(agent_action, AgentFinish):            
                # Got to the terminal state
                return True, agent_action        
            else:    
                ## Non-terminal state
                self.intermediate_steps.append((agent_action, observation))
                return False, None            

        except Exception as e:
            agent_action = AgentAction(
                log='Thought: Unexpected exception has been raised. Couldn\'t get either AgentAction or AgentFinish.\nAction:\n```\n{\n"action": "",\n"action_input": ""\n}\n```',
                tool='',
                tool_input='',
                type = 'AgentAction')
                
            Thought, _ = self._get_Thought_and_Action(agent_action.log, print_on_stdout=True)
            Action = f'Action {self.timestep+1}: ""'
            observation = f'Observation {self.timestep+1}: Unexpected Exception has been raised. Couldn\'t get either AgentAction or AgentFinish. The error message is "{e}". I should try again.+\n'
            self.print_on_stdout(observation)
            agent_log = Thought+'\n'+Action+'\n'+ observation
            self.agent_log[-1] += agent_log
            self.intermediate_steps.append((agent_action, observation))
            return False, None  
