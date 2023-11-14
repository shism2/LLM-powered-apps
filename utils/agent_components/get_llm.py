import os
from utils.load_vars import get_param
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI



def get_base_llm(config):
    try: streaming = config.streaming.value
    except AttributeError: streaming = config.streaming  
    if config.provider == 'AzureChatOpenAI':
        return AzureChatOpenAI(deployment_name=get_param('deployment_name'), temperature=0.0, streaming=streaming)
    elif config.provider == 'ChatOpenAI':
        return ChatOpenAI(temperature=0.0, streaming=streaming, model_kwargs={'engine':get_param('model_name')})
