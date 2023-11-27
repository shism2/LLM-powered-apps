import os, sys
sys.path.append('../..')
from dotenv import load_dotenv
_ = load_dotenv('../../.env')

import ast  
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain.retrievers.you import YouRetriever
from langchain.chains import RetrievalQA
from reasoning_engines.langchain_llm_wrappers import AzureChatOpenAIWrapper

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



class MyRetrievalQA(RetrievalQA):
    def run(self, query: str)-> str:
        return super().run(get_query_from_string(query)) 
    def arun(self, query: str)-> str:
        return super().arun(get_query_from_string(query)) 

def ydc_qa_chain(chain_type='stuff', llm: Optional=None)-> RetrievalQA:
    yr = YouRetriever()
    llm = llm or AzureChatOpenAIWrapper()
    RetrievalQA_chain = MyRetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=yr)
    return RetrievalQA_chain

class YDCSearch(BaseModel):
    """Get Google-search results"""
    query : str = Field(description='query for searching on Google.com') 


class YDCSearchTool:
    # For backward compatibility
    name = 'YDCSearch'
    description="Get Google-search results"
    args_schema : Type[YDCSearch] = YDCSearch
    
    def __init__(self, chain_type: str='stuff', llm: Optional=None):
        self.chain_type = chain_type
        self.llm = llm
        self.qa_chain = ydc_qa_chain(chain_type=self.chain_type, llm=self.llm)

    def run(self, query: str)-> str:
        return self.qa_chain(query)['result']

    async def get_search_result(self, query: str)-> str:
        return self.run(query) 

    async def arun(self, query: str)-> str:
        result = await self.get_search_result(query)
        return result

def get_YDCSearch_schema_and_tool(chain_type='stuff', llm: Optional=None):
    return YDCSearch, YDCSearchTool(chain_type=chain_type, llm=llm)
