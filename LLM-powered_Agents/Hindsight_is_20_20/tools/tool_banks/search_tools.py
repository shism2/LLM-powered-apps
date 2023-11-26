import os, sys
sys.path.append('../..')
from dotenv import load_dotenv
_ = load_dotenv('../../.env')

import ast  
from typing import Type
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

def ydc_qa_chain(chain_type='stuff')-> RetrievalQA:
    yr = YouRetriever()
    llm = AzureChatOpenAIWrapper()
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
    
    def __init__(self, chain_type: str='stuff'):
        self.chain_type = chain_type
        self.qa_chain = ydc_qa_chain(self.chain_type)

    def run(self, query: str)-> str:
        return self.qa_chain(query)['result']

    async def get_search_result(self, query: str)-> str:
        return self.run(query) 

    async def arun(self, query: str)-> str:
        result = await self.get_search_result(query)
        return result

def get_YDCSearch_schema_and_tool(chain_type='stuff'):
    return YDCSearch, ('YDCSearch', YDCSearchTool(chain_type))
