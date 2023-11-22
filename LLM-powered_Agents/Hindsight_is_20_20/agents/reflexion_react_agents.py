from agents.base_cumtom_agent import BaseCustomAgent
from agents.react_agents import ReActAgent
from typing import Any, List, Tuple, Optional
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from typing import Literal
from datetime import datetime
import pytz

class ReflexionReActAgent(ReActAgent):
    def __init__(self, 
                reflexion_chain: Any,
                reflexion_header: str, 
                **kwargs):
        super().__init__(**kwargs)

        self.reflexion_chain = reflexion_chain
        self.reflexion_header = reflexion_header

        self.reflexion = ''+self.reflexion_header
        self.last_reflexion = None

    #############################
    #### Fundamental methods ####
    #############################
    def get_log_prefix(self):
        korea_time = datetime.now(pytz.timezone('Asia/Seoul'))  
        year, month, day = korea_time.year, korea_time.month, korea_time.day
        return f'ReflexionReAct_{year}-{month}-{day}'




    ####################################
    #### Agentic simulation methods ####
    ####################################
    def agent_step(self, query: str)-> Tuple[bool, AgentFinish|None]:
        self._before_agent_step()
        try:
            agent_action = self.brain.invoke({
                'intermediate_steps': self.intermediate_steps,
                'input': query,
                'reflections':self.reflexion, 
            })
        except Exception as e:
            """ This catches the exception where the brain fails to produce AgentAction or AgentFinish.  """
            agent_action = AgentAction(
                log='Thought: Unexpected exception has been raised. Brain cannot produce AgentAction or AgentFinish. ' + f'The error message is "{e}".'+ '\nAction:\n```\n{\n"action": "",\n"action_input": ""\n}\n```',
                tool='',
                tool_input='',
                type = 'AgentAction')
        finally:
            Observation, Thought_Action_Observation = self.func_execution(agent_action=agent_action)
            self.agent_log += Thought_Action_Observation

            if isinstance(agent_action, AgentFinish):            
                return True, agent_action        
            else:    
                self.intermediate_steps.append((agent_action, Observation))
                return False, None            



    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None)-> None:
        if reference==None:            
            raise ValueError("For Reflexion agent, reference should be provided for 'run_agent_trials' method.")
        self.collect_logs(f"----- New test point -----", (False, 'info'), (True, 'info'), (True, 'info'))
        self.collect_logs(f"Query: {query}", (True, 'info'), (True, 'info'), (False, 'info'))
        self._before_agent_trials()


        trial=0
        while self.judgement[1]!='CORRECT' and trial<num_trials:
            self.collect_logs(f"Trial {trial+1}", (True, 'info'), (True, 'info'), (False, 'info'))
            if trial>0:
                    self.collect_logs(f"Reflexion......", (True, 'info'), (True, 'info'), (False, 'info'))            
                    self.do_reflexion(self.agent_log_for_trajectory)
                    reflexion_loglevel = 'info' if len(self.last_reflexion.split('I could not produce a reflexion for this trial'))==1 else 'error'
                    self.collect_logs(self.reflexion, (True, reflexion_loglevel), (True, reflexion_loglevel), (False, reflexion_loglevel))
                    self.collect_logs(self.last_reflexion, (False, reflexion_loglevel), (False, reflexion_loglevel), (True, reflexion_loglevel))
            self.run_agent_episode(query=query, reference=reference, trial=trial)  
            trial += 1








    ###########################################
    #### Agentic-simulation helper methods ####
    ###########################################
    def _before_agent_trials(self):
        self.judgement = ['', 0]
        self.reflexion_reset()




    ######################################
    #### This-Class specific methods  ####
    ######################################
    def do_reflexion(self, agent_log_for_trajectory:str)-> str:
        new_reflexion = self.reflexion_chain(agent_log_for_trajectory)
        self.reflexion += new_reflexion
        self.last_reflexion = self.reflexion_header + new_reflexion

    def reflexion_reset(self)-> None:
        self.reflexion = ''+self.reflexion_header
        self.last_reflexion = None

 







