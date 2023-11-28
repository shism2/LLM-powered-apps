from langchain.evaluation import load_evaluator
from typing import Optional
from utils.wrappers import retry
from openai import RateLimitError
from reasoning_engines.langchain_llm_wrappers import QuickAzureChatOpenAI

class CustomQAEvaluator:
    def __init__(self, llm: Optional=None):
        self.llm = llm or QuickAzureChatOpenAI()
        self.evaluator_type = 'qa'
        self.evaluator = load_evaluator(evaluator=self.evaluator_type, llm=self.llm)

    @retry(allowed_exceptions=(RateLimitError,))
    def __call__(self, query: str, reference: str, prediction: str):
        return self.evaluator.evaluate_strings(
            input = query, 
            prediction = prediction, 
            reference = reference,
        )
