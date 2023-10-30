import os
from pydantic import BaseModel, Field
from agent_enums import Search, RetriecalChainType, AzureDeploymentName, Boolean, Agentype

class Configurations(BaseModel):
    agent_type: Agentype = Field(default=Agentype.openai, description="LangChain agent type")
    azure_gpt_deployment_name: AzureDeploymentName = Field(default=AzureDeploymentName.gpt4_32k, description="Azure OpenAI LLM deployment name")
    retrieval_chain_type: RetriecalChainType = Field(default=RetriecalChainType.stuff, description="Chain type for LangChain Retrieval QA chain")
    llm_search_api_chain_type: RetriecalChainType = Field(default=RetriecalChainType.stuff, description="Chain type for LangChain Retrieval QA chain, in case search API needs Retrieval QA chain such as YDC API")
    search_tool: Search = Field(default=Search.YDC, description="Google search API to use")
    verbose: Boolean = Field(default=Boolean.true, description="verboseness for agent")
    qna_log_folder:str = Field(default='loggers/qna_logs', description="folder qna logs are stored")
    scratchpad_log_folder:str = Field(default='loggers/scratchpad_logs', description="folder agent's intermediate-step logs are stored")
    streaming:Boolean = Field(default=Boolean.true, description="whether to stream the final output or not")


def folder_existence_check(config: Configurations)-> None:
    for k, v in config:
        if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
            os.makedirs(v)  
