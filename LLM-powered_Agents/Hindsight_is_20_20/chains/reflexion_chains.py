from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import hub
from typing import List, Tuple, Any, Dict, Optional, Literal, Type
from langchain.agents import Tool
from langchain.tools.render import render_text_description_and_args
from langchain.schema.runnable import RunnableSequence
from pydantic import BaseModel, Field
from openai import RateLimitError
from utils.wrappers import retry

class BaseReflexionChain:
    def __init__(self, 
                reasoninig_engine: Any,
                prompt: ChatPromptTemplate|str,
                schemas_and_tools: Any,
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
        self.schemas_and_tools = schemas_and_tools
        self.schemas = self.schemas_and_tools.schemas()
        self.tools = self.schemas_and_tools.tools()


        # reflextion few-shot examples
        self.reflexion_examples = reflexion_examples

    def __tool_description_for_system_msg__(self, schemas: List[Type[BaseModel]], tools: List[Tool])->str:
        """adapted from langchain's render_text_description_and_args """
        tool_strings = []
        for schema, tool in zip(schemas, tools):
            tool_strings.append(f"{tool.name}: {tool.description}, args: {str(schema.schema()['properties'])}")
        return "\n".join(tool_strings)


    @property
    def prompt(self)-> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([self.system_prompt, self.human_prompt])
        prompt = prompt.partial(
            tool_description =self.__tool_description_for_system_msg__(self.schemas, self.tools),
            examples = self.reflexion_examples,
            )
        return prompt


    @property
    def reflexion_chain(self)-> RunnableSequence:
        return self.prompt | self.reasoninig_engine    


    @retry(allowed_exceptions=(RateLimitError,))
    def get_reflextion(self, previous_trial: str)->str:
        return '\n- ' + self.reflexion_chain.invoke({'previous_trial':previous_trial}).content                                  


    def __call__(self, previous_trial: str)-> str:
        try:
            reflexion = self.get_reflextion(previous_trial)
        except Exception as e:
            reflexion = '\n- ' + f'I could not produce a reflexion for this trial because of an unexpected error to my reflexion brain. The error message if {e}'
        return reflexion

