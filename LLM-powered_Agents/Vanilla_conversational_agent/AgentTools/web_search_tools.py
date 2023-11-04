import os, sys
sys.path.extend(['..', '../..'])
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.retrievers.you import YouRetriever
from typing import Type
from langchain.agents import Tool
from utils.get_llm import get_base_llm
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
from utils.load_vars import get_param
from agent_specific.configurations import Configurations
from dotenv import load_dotenv
_ = load_dotenv('.env')


'''
Tranditonal Google API tool
'''
def GetFromGoogleAPI()-> Tool:
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="Googling_from_Google",
        func=search.run,
        description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a string object."
    )
    return tool


'''
Serp API tool
https://serpapi.com/
'''
def GetFromSerpAPI()-> Tool:
    search = SerpAPIWrapper()
    tool = Tool(
        name="Googling_from_SerpAPI",
        func=search.run,
        description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a string object."
    )
    return tool


'''
Using You.com (YouRetriever)
https://serpapi.com/
'''
def ydc_qa_chain(config)-> RetrievalQA:
    yr = YouRetriever()
    llm = get_base_llm(config)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=yr)

class GetFromYDCAPIArgs(BaseModel):
    Query: str = Field(description='query for searching on Google.com')

class GetFromYDCAPI(StructuredTool):
    def __init__(self, qa_chain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qa_chain = qa_chain
    name = 'GetFromYDC'
    description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a string object."
    args_schema : Type[GetFromYDCAPIArgs] = GetFromYDCAPIArgs

    class Config:
        use_enum_values = False
        extra = 'allow'

    def _run(self, Query: str)-> str:
        return self.qa_chain.run(Query)
    async def _arun(self, Query: str)-> str:
        raise NotImplementedError("Does not support async operation yet.")
    
    
def get_web_search_tools(config)-> Tool:
    if config.search_tool.value == 'google':
        return GetFromGoogleAPI()
    if config.search_tool.value == 'Serp AIP':
        return GetFromSerpAPI()
    if config.search_tool.value == 'You.com':
        qa_chain = ydc_qa_chain(config)
        return GetFromYDCAPI(qa_chain)
    