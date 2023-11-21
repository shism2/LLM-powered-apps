from agents.base_cumtom_agent import BaseCustomAgent
from agents.react_agents import ReActAgent
from typing import Any, List, Tuple, Optional
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from typing import Literal

class ReflexionReActAgent(ReActAgent):
    def __init__(self, 
                reflexion_chain: Any,
                reflexion_header: str, 
                **kwargs):
        super().__init__(**kwargs)

        self.reflexion_chain = reflexion_chain
        self.reflexion_header = reflexion_header

        self.reflexion = ''+self.reflexion_header

    #############################
    #### Fundamental methods ####
    #############################

    ####################################
    #### Agentic simulation methods ####
    ####################################
    def execution(self, agent_action: AgentAction|AgentFinish|Literal['NO_NEED'], judgement=False)-> str|None:
        if judgement:
            self.agent_log[-1] += self.judgement[0]
            self.print_on_stdout(self.judgement[0])
            return


        Thought, Action = self._get_Thought_and_Action(agent_action.log)
        
        # Observation or Answer
        if isinstance(agent_action, AgentAction):
            try:
                observation = self.tool_dictionary[agent_action.tool].run(agent_action.tool_input)
                Observation = (f'Observation {self.timestep+1}: '+observation).rstrip('\n')
            except Exception as e:
                Observation = f'Observation {self.timestep+1}: Filed to get Observation (function output). The tool is {agent_action.tool} and tool input is {agent_action.tool_input}. The error message is "{e}"'
            finally:     
                self.print_on_stdout(Observation, sep='')
                return Observation, Thought+'\n'+Action+'\n'+Observation+'\n'
        else:
            try:
                Observation = (f'Answer: '+agent_action.return_values['output']).rstrip('\n') 
            except Exception as e:
                Observation = f'Answer: Failed to get the final answer. The error message is "{e}"'
            finally:
                self.print_on_stdout(Observation, sep='')
                return None, Thought+'\n'+Action+'\n'+Observation+'\n'



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



    def run_agent_trials(self, num_trials: int, query: str, reference: Optional[str]=None, agent_log_reset=True)-> None:
        '''
        Override this method for any child class
        '''
        if reference==None:
            raise ValueError("For Reflexion agent, reference should be provided for 'agent_run_miltiple_trials' method.")
        self.print_on_stdout(f"Query: {query}")

        if agent_log_reset:
            self.agent_log_reset()
            self.reflexion_reset()


        trial=0
        while self.judgement[1]!=1 and trial<num_trials:
            self.print_on_stdout(f"---- Trial {trial+1} ----")
            if trial>0:
                    self.print_on_stdout(f"Reflexion......")
                    self.do_reflexion(self.agent_log[-1])
                    print(f"{self.reflexion}")
            self.run_agent_episode(query=query, reference=reference, multiple_trials=True)  
            trial += 1


    ###########################################
    #### Agentic-simulation helper methods ####
    ###########################################



    ######################################
    #### This-Class specific methods  ####
    ######################################
    def do_reflexion(self, agent_log:str)-> str:
        reflexion = self.reflexion_chain(agent_log)
        self.reflexion += reflexion+'\n'

    def reflexion_reset(self)-> None:
        self.reflexion = ''+self.reflexion_header

 







