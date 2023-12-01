from typing import List
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool

# Parsing Langchain AI message into OpenAI AI message dict
def parsing_to_ai_msg_dict(chat_completion_msg):
    return {'role':'assistant', 'content': chat_completion_msg.content, 'tool_calls' : chat_completion_msg.tool_calls}

async def parsing_to_ai_msg_dict_a(chat_completion_msg):
    return parsing_to_ai_msg_dict(chat_completion_msg)

async def parsing_to_ai_msg_dict_async(chat_completion_msg):
    ai_msg_dict = await parsing_to_ai_msg_dict_a(chat_completion_msg)
    return ai_msg_dict

# Parsing tool-invocation result and meta data into OpenAI tool message dict
def parsing_to_tool_msg_dict(tool_responses: List)-> List:
    return [{'role':'tool', 'name':tool_response[1], 'tool_call_id':tool_response[0], 'content':tool_response[2]} for tool_response in tool_responses]

async def parsing_to_tool_msg_dict_a(tool_responses: List)-> List:
    return parsing_to_tool_msg_dict(tool_responses)

async def parsing_to_tool_msg_dict_async(tool_responses: List)-> List:
    result =  await parsing_to_tool_msg_dict_a(tool_responses=tool_responses) 
    return result



# Get OpenAI's functions from schemas
def get_openai_functions_from_schemas(schemas: List)-> List:
    return [convert_pydantic_to_openai_tool(x) for x in schemas]
