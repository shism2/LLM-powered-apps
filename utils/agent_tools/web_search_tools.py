import os, sys
sys.path.extend(['..', '../..'])
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.retrievers.you import YouRetriever
from typing import Type
from langchain.agents import Tool
from utils.agent_components.get_llm import get_base_llm
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
from utils.load_vars import get_param
from agent_specific.configurations import Configurations
import ast  
from dotenv import load_dotenv
_ = load_dotenv('.env')


# for robustness
def get_query_from_string(s):  
    # Check if string is a list  
    if s.startswith('[') and s.endswith(']'):  
        # Use ast.literal_eval to convert string to list  
        list_representation = ast.literal_eval(s)  
        # Get the second item from the list  
        query = list_representation[1]  
    else:  
        # If the string is not a list, return the string as is  
        query = s  
    return query  

'''
Tranditonal Google API tool
'''
class MyGoogleSearchAPIWrapper(GoogleSearchAPIWrapper):
    def run(self, Query: str)-> str:
        return super().run(get_query_from_string(Query)) 
    def arun(self, Query: str)-> str:
        return super().arun(get_query_from_string(Query)) 

def GetFromGoogleAPI()-> Tool:
    search = MyGoogleSearchAPIWrapper()
    tool = Tool(
        name="Googling_from_Google",
        func=search.run,
        description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a single string not a list. Never pass in ['string', $query], Just pass in $query."
    )
    return tool


'''
Serp API tool
https://serpapi.com/
'''
class MySerpAPIWrapper(SerpAPIWrapper):
    def run(self, Query: str)-> str:
        return super().run(get_query_from_string(Query)) 
    def arun(self, Query: str)-> str:
        return super().arun(get_query_from_string(Query)) 

def GetFromSerpAPI()-> Tool:
    search = MySerpAPIWrapper()
    tool = Tool(
        name="Googling_from_SerpAPI",
        func=search.run,
        description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a single string not a list. Never pass in ['string', $query], Just pass in $query."
    )
    return tool


'''
Using You.com (YouRetriever)
https://serpapi.com/
'''
class MyRetrievalQA(RetrievalQA):
    def run(self, Query: str)-> str:
        return super().run(get_query_from_string(Query)) 
    def arun(self, Query: str)-> str:
        return super().arun(get_query_from_string(Query)) 

def ydc_qa_chain(config)-> RetrievalQA:
    yr = YouRetriever()
    llm = get_base_llm(config)
    RetrievalQA_chain = MyRetrievalQA.from_chain_type(llm=llm, chain_type=config.retrieval_chain_type.value, retriever=yr)
    return RetrievalQA_chain

class GetFromYDCAPIArgs(BaseModel):
    Query: str = Field(description='query for searching on Google.com')

class GetFromYDCAPI(StructuredTool):
    def __init__(self, qa_chain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qa_chain = qa_chain
    name = 'GetFromYDC'
    description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a single string not a list. Never pass in ['string', $query], Just pass in $query."
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
    if config.search_tool.value == 'Serp_API':
        return GetFromSerpAPI()
    if config.search_tool.value == 'You.com':
        qa_chain = ydc_qa_chain(config)
        return GetFromYDCAPI(qa_chain)
    
