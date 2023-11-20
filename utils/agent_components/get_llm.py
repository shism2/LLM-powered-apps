import os
from utils.load_vars import get_param
from utils.agent_components.configurations import Configurations
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from typing import Optional

## This will be deprecated soon
def get_base_llm(config: Optional[Configurations]=None):
    if config == None: 
        config = Configurations()
    config_dict = config.dict()
    if config.provider == 'AzureChatOpenAI':
        llm = AzureChatOpenAI(deployment_name=get_param('azure_deployment_name'), model_name=get_param('azure_model_name'),
                                        temperature=config_dict['temperature'], max_tokens=config_dict['max_tokens'], streaming=config_dict['streaming'])
    elif config.provider == 'ChatOpenAI':
        llm = ChatOpenAI(model_name=get_param('model_name'), 
                                        temperature=0.0, streaming=config_dict['streaming'])
    return llm



class LangChainLLMWrapper:
    def __init__(self, config: Optional[Configurations]=None):
        if config == None: 
            config = Configurations()
        self.llm = get_base_llm(config)



