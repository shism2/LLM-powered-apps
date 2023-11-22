from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import hub
from typing import List, Tuple, Any, Dict, Optional, Literal
from langchain.agents import Tool
from langchain.tools.render import render_text_description_and_args
from utils.agent_components.get_llm import LangChainLLMWrapper
from langchain.schema.runnable import RunnableSequence

class BaseReflexionChain:
    def __init__(self, 
                reasoninig_engine: Any,
                prompt: ChatPromptTemplate|str,
                tools: List[Tool],
                reflexion_examples: str):
       
        # reasoning engine
        self.reasoninig_engine = reasoninig_engine

        # prompt for brain
        if isinstance(prompt, str):
            prompt = hub.pull(prompt)
        if len(prompt)!=2 or not isinstance(prompt[0], SystemMessagePromptTemplate) or not isinstance(prompt[1], HumanMessagePromptTemplate): 
            raise ValueError("Error in 'prompt'.")
        self.system_prompt = prompt[0]
        self.human_prompt = prompt[1]        

        # tools
        self.tools = tools

        # reflextion few-shot examples
        self.reflexion_examples = reflexion_examples

    @property
    def prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([self.system_prompt, self.human_prompt])
        prompt = prompt.partial(
            tool_description = self.tool_description,
            examples = self.reflexion_examples,
        )
        return prompt

    @property
    def tool_description(self)-> str:
        return render_text_description_and_args(self.tools)

    @property
    def reflexion_chain(self)-> RunnableSequence:
        return self.prompt | self.reasoninig_engine    


    def __call__(self, previous_trial: str)-> str:
        try:
            reflexion = '\n- ' + self.reflexion_chain.invoke({'previous_trial':previous_trial}).content
        except Exception as e:
            reflexion = '\n- ' + f'I could not produce a reflexion for this trial because of an unexpected error to my reflexion brain. The error message if {e}'
        return reflexion


if __name__ == '__main__':
    from utils.agent_tools.tools.get_tools import get_tool_list
    from fewshot_examples.reflections import REFLECTIONS
    from utils.agent_components.get_llm import LangChainLLMWrapper

    reasoninig_engine = LangChainLLMWrapper().llm
    prompt = "jet-taekyo-lee/experimental-reflextion"
    tools = get_tool_list()
    reflexion_examples = REFLECTIONS

    reflexion_chain = BaseReflexionChain(reasoninig_engine=reasoninig_engine, prompt=prompt, tools=tools, reflexion_examples=reflexion_examples)

