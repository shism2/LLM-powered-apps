from enum import Enum

class Agentype(Enum):
    openai = 'openai functioncall'
    react = 'zeroshot react'

class Search(Enum):
    Google = 'google'
    Serp = 'Serp AIP'
    YDC = 'You.com'

class RetriecalChainType(Enum):
    stuff = 'stuff'
    map_reduce = 'map_reduce '

class AzureDeploymentName(Enum):
    gpt4_8k = 'taekyo_gpt4_8k'
    gpt4_32k = 'taekyo_gpt4_32k'

class Boolean(Enum):
    true = True
    false = False