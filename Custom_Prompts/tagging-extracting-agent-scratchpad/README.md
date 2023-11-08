# Background
This prompt is designed to tag or extract associated thoughts, actions, and observations from [ReAct](https://react-lm.github.io/) agents, as implemented according to [LangChain's guide](https://python.langchain.com/docs/modules/agents/agent_types/structured_chat).

The prompt relies on two input variables: {task} and {scratchpad}:  
1. task : This is the query that an end user sends to the ReAct agent. Here's an example:  
```
task = "Which is bigger? BTS's oldest member's birth month vs. incumbent Korean president's birth month."
```

2. scratchpad :  This is a history of the ReAct agent's process for carrying out the task up to a certain point. It doesn't need to be the complete history with the 'Finished chain' comment. Essentially, this is **what you'd see on stdout when a ReAct agent is executed with `verbose=True`**. Here's an example (Note that this example is a complete history, which is not a requirement):
```
scratchpad = """Entering new AgentExecutor chain...
Thought: I need to find out the birth month of the oldest member of BTS and the birth month of the current Korean president. 

Action:
{
  "action": "GetFromYDC",
  "action_input": {
    "Query": "Oldest member of BTS birth month"
  }
}

The oldest member of BTS is Jin, and his birth month is December. He was born on December 4, 1992.

Action:

{
  "action": "GetFromYDC",
  "action_input": {
    "Query": "Current Korean president birth month"
  }
}

The current President of South Korea, Yoon Suk Yeol, was born on 18 December 1960. So, his birth month is December.

Action:
{
  "action": "Final Answer",
  "action_input": "The birth month of the oldest member of BTS, Jin, and the birth month of the current Korean president, Yoon Suk Yeol, are both December. Therefore, neither is bigger as they are the same."
}


Finished chain."""
```

The output generated by the chain looks like this:
```
response = chain.invoke({'task':task, 'scratchpad':scratchpad})
print(response.additional_kwargs['function_call']['arguments'])
{
  "task": "Which is bigger? BTS's oldest member's birth month vs. incumbent Korean president's birh month.",
  "thoughts": [
    {
      "thought": "I need to find out the birth month of the oldest member of BTS and the birth month of the current Korean president."
    }
  ],
  "actions": [
    {
      "action": "GetFromYDC",
      "action_input": {
        "Query": "Oldest member of BTS birth month"
      }
    },
    {
      "action": "GetFromYDC",
      "action_input": {
        "Query": "Current Korean president birth month"
      }
    },
    {
      "action": "Final Answer",
      "action_input": "The birth month of the oldest member of BTS, Jin, and the birth month of the current Korean president, Yoon Suk Yeol, are both December. Therefore, neither is bigger as they are the same."
    }
  ],
  "observations": [
    {
      "observation": "The oldest member of BTS is Jin, and his birth month is December. He was born on December 4, 1992."
    },
    {
      "observation": "The current President of South Korea, Yoon Suk Yeol, was born on 18 December 1960. So, his birth month is December."
    }
  ]
}
```

# Use Cases
This prompt is incredibly useful for extracting structured components after each iteration of the ReAct agent's task-completion process.
- You might want to extract only 'observation' components from the agent's scratchpad, or perhaps 'thought' or 'action' components. In these cases, you can parse the output into a Python dictionary and retrieve the desired components.  
- This prompt is also expected to see frequent use in other chains. For instance, if you have a chain that requires a summary of observations from the ReAct agent, this prompt can be used for the intermediate chain associated with the ReAct agent.


# OpenAI Function calling
This prompt needs OpenAI's Function calling. To test it, you need to bind the 'functions' argument to the LLM. A Jupyter notebook file is provided below for you to try it out.
Unfortunately, I found that currently, it raises an error when trying it on LangChain Hub's Prompt Playground even with the 'Function Call' parameter set. I am investigating the cause and will fix it as soon as possible. In the meantime, please try this prompt on your local machine using the provided notebook file.

# How to play around?
The following code snippet provides an example of how you can utilize the prompt with LLMs. Since this prompt is designed to work with OpenAI's Function calling as mentioned above, you should use either ChatOpenAI() or AzureChatOpenAI() as your base LLM.
```
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

trajectory_getting_func = convert_pydantic_to_openai_function(Trajectory)

def get_OpenAI_chain(provider: Literal['ChatOpenAI', 'AzureChatOpenAI'], model_or_deployment_name: str):
    if provider == 'ChatOpenAI':
        chat_model = ChatOpenAI(model_name=model_or_deployment_name, temperature=0).bind(functions=[trajectory_getting_func], function_call={'name':'Trajectory'})
    else:
        chat_model = AzureChatOpenAI(deployment_name=model_or_deployment_name, temperature=0).bind(functions=[trajectory_getting_func], function_call={'name':'Trajectory'})
    return chat_model

prompt = hub.pull("jet-taekyo-lee/tagging-extracting-agent-scratchpad")
chat_model = get_OpenAI_chain(provider, deployment_name)
chain = prompt | chat_model

response = chain.invoke({'task':task, 'scratchpad':scratchpad})
print(response.additional_kwargs['function_call']['arguments'])
```

Visit the provided [GitHub repo](https://github.com/Taekyo-Lee/LLM-powered-apps/tree/main/Custom_Prompts/tagging-extracting-agent-scratchpad) and download [tagging-extracting-agent-scratchpad/nb_tagging-extracting-agent-scratchpad.ipynb](https://github.com/Taekyo-Lee/LLM-powered-apps/blob/main/Custom_Prompts/tagging-extracting-agent-scratchpad/nb_tagging-extracting-agent-scratchpad.ipynb) file.