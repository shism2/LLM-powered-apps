import os
from utils.load_envs import Environ
from typing import List, Dict, Optional
from langchain.chat_models import AzureChatOpenAI
import time
from openai import RateLimitError
from utils.wrappers import retry
import asyncio
from abc import ABC, abstractmethod  

def QuickAzureChatOpenAI(version:str, max_retries:int=0, temperature=0.0, model_kwargs:Dict={}, **kwargs):
    _ = Environ().azure_openai(version)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv('azure_deployment'), 
        model_name=os.getenv('model_name'),
        max_retries = max_retries,
        temperature = temperature,
        model_kwargs = model_kwargs,
        **kwargs
        )    
    return llm


class QuickGPTClient(ABC):
    @abstractmethod
    def __init__(self, version: Optional[str]=None):
        pass
    
    @abstractmethod
    def reload_envs(self):
        pass

    ### api without tool use
    def get_chat_response_no_tool_use(self, messages, model, stream):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream)

    ### api with tool use
    def get_chat_response_tool_use(self, messages, model, stream, tools):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream,
                        tools = tools,
                        tool_choice = 'auto')

    ### Chat completion api entry point
    def chat_completions_create(self, query:Optional[str]=None, messages:Optional[List[Dict]]=None, tools:Optional[List]=None, return_message=True, stream=False):
        self.reload_envs()
        if messages != None:
            messages = messages
        else:
            if query != None:
                messages = [{'role':'user','content':query }]
            else:
                raise ValueError("Pass in either query or messages")
        chat_response = self.get_chat_response_tool_use(messages=messages, model=self.model, stream=stream, tools=tools) if tools !=None else self.get_chat_response_no_tool_use(messages=messages, model=self.model, stream=stream)
        return chat_response.choices[0].message if return_message else chat_response.choices[0].message.content    
            
    def __call__(self, **kwargs):
        return self.chat_completions_create(**kwargs)




class AsyncQuickGPTClient(ABC):
    @abstractmethod
    def __init__(self, version: Optional[str]=None):
        pass
    
    @abstractmethod
    async def reload_envs(self):
        pass

    ### api without tool use
    async def get_chat_response_no_tool_use(self, messages, model, stream):
        response = await self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream)
        return response

    ### api with tool use
    async def get_chat_response_tool_use(self, messages, model, stream, tools):
        response = await self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream,
                        tools = tools,
                        tool_choice = 'auto')
        return response

    ### Chat completion api entry point
    async def chat_completions_create(self, query:Optional[str]=None, messages:Optional[List[Dict]]=None, tools: Optional[List]=None, return_message=True, stream=False):
        await self.reload_envs()
        if messages != None:
            messages = messages
        else:
            if query != None:
                messages = [{'role':'user','content':query }]
            else:
                raise ValueError("Pass in either query or messages")
        
        chat_response = await self.get_chat_response_tool_use(messages=messages, model=self.model, stream=stream, tools=tools) if tools !=None else await self.get_chat_response_no_tool_use(messages=messages, model=self.model, stream=stream)
        return chat_response.choices[0].message if return_message else chat_response.choices[0].message.content    
            
    async def __call__(self, **kwargs):
        response = await self.chat_completions_create(**kwargs)
        return response



from openai import OpenAI, AsyncOpenAI
class QuickOpenAIClient(QuickGPTClient):
    def __init__(self, version: Optional[str]=None):
        self.env = Environ()
        self.env.openai()
        self.client = OpenAI()
        self.model = 'gpt-4-1106-preview'

    def reload_envs(self):
        self.env.openai()
        self.model = 'gpt-4-1106-preview'    

class AsyncQuickOpenAIClient(AsyncQuickGPTClient):
    def __init__(self, version: Optional[str]=None):
        self.env = Environ()
        self.env.openai()
        self.client = AsyncOpenAI()
        self.model = 'gpt-4-1106-preview'

    async def reload_envs(self):
        await self.env.openai_async()
        self.model = 'gpt-4-1106-preview'    




from openai import AzureOpenAI, AsyncAzureOpenAI
class QuickAzureOpenAIClient(QuickGPTClient):
    def __init__(self, version: Optional[str]=None):
        if version == None:
            raise ValueError("GPT version is necessary")
        self.version = version
        _ = Environ().azure_openai(self.version)
        self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("OPENAI_API_KEY"),  
                api_version=os.getenv("OPENAI_API_VERSION")
        )
        self.model = os.getenv("azure_deployment")

    def reload_envs(self):
        _ = Environ().azure_openai(self.version)
        self.model = os.getenv("azure_deployment")


class AsyncQuickAzureOpenAIClient(AsyncQuickGPTClient):
    def __init__(self, version: Optional[str]=None):
        if version == None:
            raise ValueError("GPT version is necessary")
        self.version = version
        _ = Environ().azure_openai(self.version)
        self.client = AsyncAzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("OPENAI_API_KEY"),  
                api_version=os.getenv("OPENAI_API_VERSION")
        )
        self.model = os.getenv("azure_deployment")

    async def reload_envs(self):
        _ = await Environ().azure_openai_async(self.version)
        self.model = os.getenv("azure_deployment")
    

