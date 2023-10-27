import os
import load_envs
from user_persona import get_random_user_persona
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from custom_output_parsers import LastSentenceOutputParser


if __name__ == '__main__':
    user_persona = get_random_user_persona().json()
    llm = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")) 
    parser = LastSentenceOutputParser(last_sentence="## Final Dialogue Line for AI Assistant")
    with open('templete.txt', mode='r') as f:
        prompt = PromptTemplate.from_template(f.read())
    # chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    chain = LLMChain(llm=llm, prompt=prompt)
    dialog_line = chain.invoke(user_persona)
    print(dialog_line['text'])


