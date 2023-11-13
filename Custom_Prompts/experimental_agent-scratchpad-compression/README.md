# Background
This custom prompt is designed to generate a succinct summary of an AgentAction object's log and observation. It uses two input variables: {log} and {observation}::

 - log : This is the log message received when invoking an agent. For a ReAct agent, it consists of Thought and Action.  
```
log = 'Thought: The user wants to know the current time in New York City and the weather forecast for the next 3 days. I need to use the GetFromDatetimeModule tool to get the current time and the GetFromOpenWeatherMap tool to get the weather forecast. The current time is necessary to provide an accurate response. 

Action:
{
"action": "GetFromDatetimeModule",
"action_input": {"IANA_timezone": "America/New_York"}
}'
```

 - observation : This is the return string from the tool associated with the agent's action.
```
observation = 'timezone: America/New_York, current time(year-month-day-hour-minute): 2023-11-12-7-18'
```

The custom prompt is designed to provide a compact summary of the log and observation.

# Use Cases
Agentic applications are crucial in the realm of LLM applications. They involve iterative execution of the internal agent. Each result string (reasoning step and result of tool execution; collectively often called 'agent scratchpad' in LangChain) is fed back into the agent at every iteration. While this process allows the agent to reason effectively, it can also lead to lengthy agent scratchpads that not only confuse the agent but also increase the prompt cost.
This custom prompt addresses this issue by extracting only the necessary summary from the result string, enabling the agent to reason as effectively as if it was fed the full one. An LLM implements this by being responsible for summarizing the results.

# How to play around?
The following code snippet provides an example of how you can utilize the prompt with LLMs. 
```
from langchain import hub
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

prompt = hub.pull("jet-taekyo-lee/agent-scratchpad-compression")
deployment_name = os.getenv('deployment_name')


chat_model = AzureChatOpenAI(deployment_name=deployment_name, temperature=0) if os.getenv('OPENAI_API_TYPE') == 'azure' else ChatOpenAI(model_name=model_name, temperature=0)
agent_scratchpad_compression_chain = prompt | chat_model

agent_action_log = '''Thought: The user wants to know the current time in New York City and the weather forecast for the next 3 days. I need to use the GetFromDatetimeModule tool to get the current time and the GetFromOpenWeatherMap tool to get the weather forecast. The current time is necessary to provide an accurate response. 
Action:
{
"action": "GetFromDatetimeModule",
"action_input": {"IANA_timezone": "America/New_York"}
}
'''
observation = 'timezone: America/New_York, current time(year-month-day-hour-minute): 2023-11-12-7-18'
comprerssed_agent_scratchpad = agent_scratchpad_compression_chain.invoke({'log':agent_action_log, 'observation':observation})

print(comprerssed_agent_scratchpad.content)
# "The current time in New York City is 7:18 on November 12th, 2023."
```

Visit the provided [GitHub repo](https://github.com/Taekyo-Lee/LLM-powered-apps/tree/main/Custom_Prompts/experimental_agent-scratchpad-compression) and download [experimental_agent-scratchpad-compression/nb_agent-scratchpad-compression.ipynb](https://github.com/Taekyo-Lee/LLM-powered-apps/blob/main/Custom_Prompts/experimental_agent-scratchpad-compression/nb_agent-scratchpad-compression.ipynb) file.