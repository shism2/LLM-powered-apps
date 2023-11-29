from langchain.evaluation import load_evaluator
from typing import Optional
from utils.wrappers import retry
from openai import RateLimitError
from reasoning_engines.langchain_llm_wrappers import QuickAzureChatOpenAI

class CustomQAEvaluator:
    def __init__(self, llm: Optional=None, azure_gpt_version: Optional[str]=None):
        if llm==None and azure_gpt_version==None:
            raise ValueError("Either llm or azure_gpt_version should be provided.")
        self.llm = llm or QuickAzureChatOpenAI(version=azure_gpt_version)
        self.evaluator_type = 'qa'
        self.evaluator = load_evaluator(evaluator=self.evaluator_type, llm=self.llm)

    @retry(allowed_exceptions=(RateLimitError,))
    def __call__(self, query: str, reference: str, prediction: str):
        return self.evaluator.evaluate_strings(
            input = query, 
            prediction = prediction, 
            reference = reference,
        )
