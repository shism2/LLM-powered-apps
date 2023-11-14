import os
from typing import Literal
from pydantic import BaseModel, Field, root_validator
from utils.agent_components.agent_enums import Search, RetrievalChainType, AzureDeploymentName, Boolean, Agentype

class Configurations(BaseModel):
    agent_type: Agentype = Field(default=Agentype.openai, description="LangChain agent type")
    retrieval_chain_type: RetrievalChainType = Field(default=RetrievalChainType.stuff, description="Chain type for LangChain Retrieval QA chain")
    api_retrieval_chain_type: RetrievalChainType = Field(default=RetrievalChainType.stuff, description="Chain type for LangChain Retrieval QA chain, in case search API needs Retrieval QA chain such as YDC API")
    search_tool: Search = Field(default=Search.Serp, description="Google search API to use")
    verbose: Boolean = Field(default=Boolean.true, description="verboseness for agent")
    qna_log_folder:str = Field(default='loggers/qna_logs', description="folder qna logs are stored")
    scratchpad_log_folder:str = Field(default='loggers/scratchpad_logs', description="folder agent's intermediate-step logs are stored")
    streaming:Boolean = Field(default=Boolean.true, description="whether to stream the final output or not")
    provider: Literal['AzureChatOpenAI', 'ChatOpenAI'] = Field(default='AzureChatOpenAI', description="provider for foundation llm")

    @root_validator
    def folder_existence_check(cls, values):
        for k, v in values.items():
            if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
                os.makedirs(v)  
        return values

    class Config:  
        use_enum_values = True

def get_agent_type_enum(agent_type: str)-> Agentype:
        if agent_type == 'OpenAI_Functions': 
            agent_type_enum = Agentype.openai
        elif agent_type == 'ReAct': 
            agent_type_enum = Agentype.react    
        elif agent_type == 'ReAct_RAG_style': 
            agent_type_enum = Agentype.react_rag_style   
        return agent_type_enum
