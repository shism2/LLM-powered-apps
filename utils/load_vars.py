import os
import pickle  
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, List
# from dotenv import load_dotenv
# _ = load_dotenv('.env')


class ChatOpenAI_vars(BaseModel):
    Env_OPENAI_API_TYPE: str = Field(default='OPENAI_API_TYPE') 
    Env_OPENAI_API_KEY: str = Field(default='OPENAI_API_KEY') 
    Param_model_name: str = Field(default='model_name')
class AzureChatOpenAI_vars(BaseModel):
    Env_OPENAI_API_TYPE: str =  Field(default='AZURE_OPENAI_API_TYPE')
    Env_OPENAI_API_VERSION: str =  Field(default='AZURE_OPENAI_API_VERSION')
    Env_OPENAI_API_BASE: str =  Field(default='AZURE_OPENAI_API_BASE')
    Env_OPENAI_API_KEY: str =  Field(default='AZURE_OPENAI_API_KEY')
    Param_deployment_name: str = Field(default='azure_deployment_name')
    Param_embedding_deployment_name: str = Field(default='azure_embedding_deployment_name')
    Param_model_name: str = Field(default='azure_model_name')




is_llm_env = lambda x : True if x.split('_')[0] == 'Env' else False
def set_llm_vars(pydantic_model: Literal[ChatOpenAI_vars, AzureChatOpenAI_vars])-> None:
    pydantic_model_dict = pydantic_model.dict()
    for k in pydantic_model_dict.keys():
        if is_llm_env(k):
            os.environ[k.split('Env_')[-1]] = os.getenv(pydantic_model_dict[k])


def overwrite_llm_envs(provider: Literal['ChatOpenAI', 'AzureChatOpenAI'])-> None:
    if provider == 'ChatOpenAI':
        pydantic_model = ChatOpenAI_vars()
    elif provider == 'AzureChatOpenAI':
        pydantic_model = AzureChatOpenAI_vars()
    set_llm_vars(pydantic_model)



def get_param(param:str)-> str:
    if os.getenv(param):
        return os.getenv(param)
    else:
        raise FileNotFoundError(f"No such a parameter '{param}' in .env file.")




# def save_and_load_api_names(flag: Optional[List]= None):
#     if isinstance(flag, List):
#         with open('usable_apis.pkl', 'wb') as f:  
#             pickle.dump(flag, f)  
#     else:
#         with open('usable_apis.pkl', 'rb') as f:  
#             return pickle.load(f)



# def set_vars(pydantic_model: Literal[ChatOpenAI_vars, AzureChatOpenAI_vars])-> None:
#     pydantic_model_dict = pydantic_model.dict()
#     for k in pydantic_model_dict.keys():
#         if is_env(k):
#             print(os.getenv(pydantic_model_dict[k]))  




