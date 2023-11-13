from typing import List, Any 
from langchain.agents import Tool
from langchain.schema.runnable.base import RunnableSequence
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate)
from langchain.schema.runnable import RunnablePassthrough
import copy

### For OpenAI Function agent
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

### For ReAct agent
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from utils.custom_output_parsers.agent_intermediatte_steps import format_intersteps_to_compressed_scratchpad
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

def get_OpenAI_Functions_agent(llm: Any, tools: List[Tool], system_msg_file:str ='../../utils/agents/openai_function_call_system_msg.txt')-> RunnableSequence:
    functions = [format_tool_to_openai_function(f) for f in tools]
    llm_with_functions = llm.bind(functions=functions)

    with open(system_msg_file, 'r') as f:
        systmem_msg_template = f.read()
    prompt = ChatPromptTemplate.from_messages([                         
            SystemMessagePromptTemplate.from_template(systmem_msg_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
    agent = RunnablePassthrough.assign(
            agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | llm_with_functions | OpenAIFunctionsAgentOutputParser()
    return agent 


def get_ReAct_agents(llm: Any, tools: List[Tool], RAG_style: bool=False, hub_url: str|None=None):
    if hub_url==None:
        obj = hub.pull("jet-taekyo-lee/rag-style-react") if RAG_style else hub.pull("jet-taekyo-lee/time-aware-react-multi-input-json")   
    else:     
        obj = hub.pull(hub_url)  
    prompt = ChatPromptTemplate.from_messages([
        obj.messages[0],
        MessagesPlaceholder(variable_name="chat_history"),    
        obj.messages[1],])
    prompt = prompt.partial(
            tools=render_text_description_and_args(tools),
            tool_names=", ".join([t.name for t in tools]),)
    if RAG_style:
        agent_scratchpad_compression_prompt = hub.pull("jet-taekyo-lee/agent-scratchpad-compression")
        scratchpad_compression_llm = copy.deepcopy(llm)
        scratchpad_compression_chain = agent_scratchpad_compression_prompt | scratchpad_compression_llm
        agent = ( RunnablePassthrough.assign( 
                        compressed_agent_scratchpad  = lambda x: format_intersteps_to_compressed_scratchpad(x["intermediate_steps"], scratchpad_compression_chain),             
                    )
                | prompt | llm | JSONAgentOutputParser() )
    else:
        llm_with_stop = llm.bind(stop=["Observation"])
        agent = (   RunnablePassthrough.assign( 
                        agent_scratchpad  = lambda x: format_log_to_str(x["intermediate_steps"]),             
                    )                
                | prompt | llm_with_stop | JSONAgentOutputParser()
        )
    return agent 