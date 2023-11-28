import os
from dotenv import load_dotenv
from typing import Optional


class Environ:    
    def __init__(self, azure_version: Optional[str]=None):
        self.openai_path = '/home/taekyolee/workspace/LLM-powered-apps/envs/openai/.env'
        if azure_version != None:
            self.azure_openai(str(azure_version))
    
    def comment(self, input: str):
        print(f"Environment variales for '{input}' are successfully registered.")

    def azure_path(self, version: str):
        return f'/home/taekyolee/workspace/LLM-powered-apps/envs/azure/{str(version)}/.env'

    def openai(self, print_sucess_comment=False):
        if os.path.exists(self.openai_path):
            load_dotenv(self.openai_path, override=True)
        else:
            raise FileNotFoundError(self.openai_path) 
        if print_sucess_comment:
            self.comment('OPEN AI')

    def azure_openai(self, version: str, print_sucess_comment=False):
        if os.path.exists(self.azure_path(version)):
            load_dotenv(self.azure_path(version), override=True)
        else:    
            raise FileNotFoundError(self.azure_path(version)) 
        if print_sucess_comment:
            self.comment(f'Azure (version {str(version)})')


