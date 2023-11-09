from enum import Enum

class Agentype(Enum):
    openai = 'OpenAI_Functions'
    react = 'ReAct'

class Search(Enum):
    Google = 'google'
    Serp = 'Serp_API'
    YDC = 'You.com'

class RetrievalChainType(Enum):
    stuff = 'stuff'
    map_reduce = 'map_reduce '

class AzureDeploymentName(Enum):
    gpt4_8k = 'taekyo_gpt4_8k'
    gpt4_32k = 'taekyo_gpt4_32k'

class Boolean(Enum):
    true = True
    false = False