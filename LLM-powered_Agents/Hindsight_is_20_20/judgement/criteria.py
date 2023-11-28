import re
import string
from langchain.evaluation import load_evaluator
from utils.agent_components.get_llm import LangChainLLMWrapper
from utils.agent_components.configurations import Configurations
from typing import Optional
from utils.wrappers import retry
from openai import RateLimitError

def normalize_answer(answer: str)-> None:
    '''
    Note: This implementation is based on [normalize_answer()] by [Noah Shinn et al.],
    sourced from the GitHub repository at [https://github.com/noahshinn024/reflexion/blob/main/hotpotqa_runs/agents.py].
    It has been used here JUST AS IS.
    '''
    def remove_articles(text):                                                        
        return re.sub(r"\b(a|an|the)\b", " ", text)
  
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def EM(answer, key: str) -> bool:
    '''
    Note: This implementation is based on [EM()] by [Noah Shinn et al.],
    sourced from the GitHub repository at [https://github.com/noahshinn024/reflexion/blob/main/hotpotqa_runs/agents.py].
    It has been used here JUST AS IS.
    '''
    return normalize_answer(answer) == normalize_answer(key)


class QA_Evaluator:
    def __init__(self, llm: Optional[LangChainLLMWrapper]=None):
        self.llm = llm if llm != None else LangChainLLMWrapper(Configurations()).llm
        self.evaluator_type = 'qa'
        self.evaluator = load_evaluator(evaluator=self.evaluator_type, llm=self.llm)

    @retry(allowed_exceptions=(RateLimitError,))
    def __call__(self, query: str, reference: str, prediction: str):
        return self.evaluator.evaluate_strings(
            input = query, 
            prediction = prediction, 
            reference = reference,
        )

