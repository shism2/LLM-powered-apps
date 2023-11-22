import os
from typing import Literal
from pydantic import BaseModel, Field, root_validator
from enum import Enum

class Boolean(Enum):
    true = True
    false = False

class HD22Configuration(BaseModel):
    provider: Literal['AzureChatOpenAI', 'ChatOpenAI'] = Field(default='AzureChatOpenAI', description="provider for foundation llm")
    temperature: float = Field(default=0.0, description="temperature for llm")
    max_tokens: int|None = Field(default=None, description="maximum completion token for llm")    
    streaming: Boolean = Field(default=Boolean.true, description="whether to stream the final output or not")
    e2e_log_folder:str = Field(default='loggers/e2e_logs', description="folder e2e logs are stored")
    trajectory_only_log_folder:str = Field(default='loggers/trajectory_only_logs', description="folder trajectory logs are stored")


    @root_validator
    def folder_existence_check(cls, values):
        for k, v in values.items():
            if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
                os.makedirs(v)  
            if k == 'temperature':
                if values[k]<0.0 or values[k]>1.0:
                    raise ValueError("Parameter 'temperature' must be between 0 and 1.")  
            if k == 'max_tokens':
                if values[k]!=None and values[k]<1:
                    raise ValueError("Parameter 'max_tokens' must be positive integer.")
        return values

    class Config:  
        use_enum_values = True