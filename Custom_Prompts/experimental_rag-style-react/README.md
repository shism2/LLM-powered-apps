# Background
This custom prompt is specifically designed for a ReAct agent that utilizes a 'compressed' agent scratchpad, as opposed to the conventional agent scratchpad. The compressed agent scratchpad can be obtained using the [agent-scratchpad-compression prompt](https://smith.langchain.com/hub/jet-taekyo-lee/agent-scratchpad-compression?organizationId=99c107dc-9be4-5ebc-9aee-41fafaa9d426). The key difference between this custom prompt and the conventional ReAct prompt is its human-message template, which incorporates a summary of the agent_scratchpad ('compressed' agent scratchpad). This summary acts as additional context for the agent's decision-making process, hence the name of this prompt.


# Use Cases
Agent applications are a fundamental aspect of LLM applications. They revolve around the iterative execution of an internal agent. Each resulting string, which includes the reasoning steps and the results of tool execution (commonly referred to as 'agent scratchpad' in LangChain), is continually fed back into the agent. While this process enhances the agent's reasoning capabilities, it can also lead to lengthy agent scratchpads. These can not only cause confusion for the agent, but also increase the cost of the prompt.

This custom prompt addresses this issue by inputting the summary of the agent scratchpad into the agent instead of the entire scratchpad, allowing the agent to view it as additional context for performing the given task. This approach maintains the agent's reasoning efficiency while reducing the complexity and cost of the prompt.


# How to play around?
The following code snippet provides an example of how you can utilize the prompt with LLMs. 
```
from langchain import hub
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from utils.custom_output_parsers.agent_intermediatte_steps import format_intersteps_to_compressed_scratchpad
from langchain.agents.output_parsers import JSONAgentOutputParser

prompt = hub.pull("jet-taekyo-lee/rag-style-react")
deployment_name = os.getenv('deployment_name')


chat_model = AzureChatOpenAI(deployment_name=deployment_name, temperature=0) if os.getenv('OPENAI_API_TYPE') == 'azure' else ChatOpenAI(model_name=model_name, temperature=0)
rag_style_react_agent = ( RunnablePassthrough.assign( compressed_agent_scratchpad = lambda x : format_intersteps_to_compressed_scratchpad(x["intermediate_steps"], agent_scratchpad_compression_chain) )
    | prompt | llm | JSONAgentOutputParser()
)

from langchain.agents import AgentExecutor
rag_style_react_agent_executor = AgentExecutor(agent=rag_style_react_agent, tools=tools, verbose=True)

rag_style_react_agent_executor.invoke({'input':'What is the weather in new york?'})
```

Visit the provided [GitHub repo](https://github.com/Taekyo-Lee/LLM-powered-apps/tree/main/Custom_Prompts/experimental_rag-style-react) and download [experimental_rag-style-react/nb_experimental_rag-style-react.ipynb](https://github.com/Taekyo-Lee/LLM-powered-apps/blob/main/Custom_Prompts/experimental_rag-style-react/nb_experimental_rag-style-react.ipynb) file.