import os
from utils.load_envs import Environ
from typing import List, Dict, Optional
from langchain.chat_models import AzureChatOpenAI
import time
from openai import RateLimitError
from utils.wrappers import retry


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



from openai import OpenAI
class QuickOpenAIClient():
    def __init__(self):
        self.env = Environ()
        self.env.openai()
        self.client = OpenAI()
        self.model = 'gpt-4-1106-preview'

    def reload_envs(self):
        self.env.openai()
        self.model = 'gpt-4-1106-preview'
    
    @retry(allowed_exceptions=(RateLimitError,))
    def get_chat_response_no_tool_use(self, messages, model, stream):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream)

    @retry(allowed_exceptions=(RateLimitError,))
    def get_chat_response_tool_use(self, messages, model, stream, tools):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream,
                        tools = tools,
                        tool_choice = 'auto')


    def chat_completions_create(self, query:Optional[str]=None, messages:Optional[List[Dict]]=None, tools:Optional[List]=None, return_message=True, stream=False):
        if messages != None:
            messages = messages
        else:
            if query != None:
                messages = [{'role':'user','content':query }]
            else:
                raise ValueError("Pass in either query or messages")
        self.reload_envs()
        chat_response = self.get_chat_response_tool_use(messages=messages, model=self.model, stream=stream, tools=tools) if tools !=None else self.get_chat_response_no_tool_use(messages=messages, model=self.model, stream=stream)
        return chat_response.choices[0].message if return_message else chat_response.choices[0].message.content        

    def __call__(self, **kwargs):
        return self.chat_completions_create(**kwargs)


from openai import AzureOpenAI
class QuickAzureOpenAIClient():
    def __init__(self, version):
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
    
    @retry(allowed_exceptions=(RateLimitError,))
    def get_chat_response_no_tool_use(self, messages, model, stream):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream)

    @retry(allowed_exceptions=(RateLimitError,))
    def get_chat_response_tool_use(self, messages, model, stream, tools):
        return self.client.with_options(max_retries=0).chat.completions.create(
                        messages=messages,
                        model=self.model,
                        stream=stream,
                        tools = tools,
                        tool_choice = 'auto')



    def chat_completions_create(self, query:Optional[str]=None, messages:Optional[List[Dict]]=None, tools:Optional[List]=None, return_message=True, stream=False):
        if messages != None:
            messages = messages
        else:
            if query != None:
                messages = [{'role':'user','content':query }]
            else:
                raise ValueError("Pass in either query or messages")
        self.reload_envs()
        chat_response = self.get_chat_response_tool_use(messages=messages, model=self.model, stream=stream, tools=tools) if tools !=None else self.get_chat_response_no_tool_use(messages=messages, model=self.model, stream=stream)
        return chat_response.choices[0].message if return_message else chat_response.choices[0].message.content     

    def __call__(self, **kwargs):
        return self.chat_completions_create(**kwargs)