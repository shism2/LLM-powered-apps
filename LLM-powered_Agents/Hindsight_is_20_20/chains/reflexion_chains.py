from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import hub
from typing import List, Tuple, Any, Dict, Optional, Literal
from langchain.agents import Tool
from langchain.tools.render import render_text_description_and_args
from utils.agent_components.get_llm import LangChainLLMWrapper

class BaseReflexionChain:
    def __init__(self, 
                llm: Any,
                prompt: ChatPromptTemplate|str,
                tools: List[Tool],
                reflexion_examples: str):
       
        # reasoning engine
        self.llm = llm

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
    def reflexion_chain(self)-> Any:
        return self.prompt | self.llm    


    def __call__(self, previous_trial: str)-> str:
        return '- ' + self.reflexion_chain.invoke({'previous_trial':previous_trial}).content


if __name__ == '__main__':
    from utils.agent_tools.tools.get_tools import get_tool_list
    from fewshot_examples.reflections import REFLECTIONS
    from utils.agent_components.get_llm import LangChainLLMWrapper

    llm = LangChainLLMWrapper().llm
    prompt = "jet-taekyo-lee/experimental-reflextion"
    tools = get_tool_list()
    reflexion_examples = REFLECTIONS

    reflexion_chain = BaseReflexionChain(llm=llm, prompt=prompt, tools=tools, reflexion_examples=reflexion_examples)

