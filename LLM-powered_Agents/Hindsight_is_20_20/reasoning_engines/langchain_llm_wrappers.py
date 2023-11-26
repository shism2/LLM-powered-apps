import os, sys
from dotenv import load_dotenv
sys.path.append('../..')
_ = load_dotenv('../../.env')
from typing import Optional, Dict
from utils.load_vars import overwrite_llm_envs
from langchain.chat_models import AzureChatOpenAI

def AzureChatOpenAIWrapper(model_kwargs:Dict={}, **kwargs):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv('azure_deployment_name'), 
        model_name=os.getenv('azure_model_name'),
        model_kwargs = model_kwargs,
        **kwargs
        )    
    return llm