import os
from utils.load_vars import get_param
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI



def get_base_llm(config):
    if config.provider == 'AzureChatOpenAI':
        return AzureChatOpenAI(deployment_name=get_param('deployment_name'), temperature=0.0, streaming=config.streaming.value)
    elif config.provider == 'ChatOpenAI':
        return ChatOpenAI(temperature=0.0, streaming=config.streaming.value, model_kwargs={'engine':get_param('model_name')})
