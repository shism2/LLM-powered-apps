import os
import load_envs
from langchain.utilities import SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.retrievers.you import YouRetriever
from typing import Type
from langchain.agents import Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool


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
def ydc_qa_chain()-> RetrievalQA:
    yr = YouRetriever()
    LLM_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_LLM_DEPLOYMENT_NAME')
    llm = AzureChatOpenAI(deployment_name=LLM_DEPLOYMENT_NAME)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=yr)

class GetFromYDCAPIArgs(BaseModel):
    Query: str = Field(description='query for searching on Google.com')

class GetFromYDCAPI(StructuredTool):
    name = 'GetFromYDC'
    description="This is useful for when you need to search Google to answer questions. You should ask targeted questions. Input should be a string object."
    args_schema : Type[GetFromYDCAPIArgs] = GetFromYDCAPIArgs
    qa_chain = ydc_qa_chain()

    def _run(self, Query: str)-> str:
        return self.qa_chain.run(Query)
    async def _arun(self, Query: str)-> str:
        raise NotImplementedError("Does not support async operation yet.")
    


def get_web_search_tools(search_tool:str)-> Tool:
    if search_tool == 'google':
        return GetFromGoogleAPI()
    if search_tool == 'Serp AIP':
        return GetFromSerpAPI()
    if search_tool == 'You.com':
        return GetFromYDCAPI()
    