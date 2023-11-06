from pydantic import BaseModel, Field
from typing import List

class Thought(BaseModel):
    """Represents the agent's thought process or strategy for addressing the task."""
    thought: str = Field(description="The agent's plan or approach to get the given task completed.")

class Action(BaseModel):
    """Details the specific action taken by the agent."""
    action: str = Field(description="The specific action executed by the agent, either 'Final Answer' or a function name.")
    action_input: str = Field(description="The inputs chosen by the agent if the action is not 'Final answer'; otherwise, it's the final answer provided to the end-user.")

class Observation(BaseModel):
    """Captures the agent's observation of the current state or environment."""
    observation: str =  Field(description="The agent's insights or observations regarding the current state after executing an action. After every action which is not 'Final Answer', there must be an observation. Without an action which is not 'Final Answer', there must be no observation either.")

class Trajectory(BaseModel):
    """A comprehensive model encapsulating the task, associated thoughts, actions, and observations."""
    task: str = Field(description="The specific task assigned to the agent.")
    thoughts: List[Thought] = Field(description="A sequential record of the agent's thoughts, displaying the reasoning process.")
    actions: List[Action] = Field(description="A sequential list of actions taken by the agent in response to the task.")
    observations: List[Observation] = Field(description="A series of observations made by the agent throughout the task execution.")

