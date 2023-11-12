from typing import List, Tuple
from langchain.schema.agent import AgentAction
from langchain.schema.runnable.base import RunnableSequence

def format_intersteps_to_compressed_scratchpad(intermediate_steps: List[Tuple[AgentAction, str]], agent_scratchpad_compression_chain: RunnableSequence) -> str:
    """from langchain.schema.agent.format_log_to_str"""
    extraction = ""
    for i, (agent_action, observation) in enumerate(intermediate_steps):
        extraction +=  '- ' + (agent_scratchpad_compression_chain.invoke({'log':agent_action.log, 'observation':observation})).content + '\n'   
    return extraction