from langchain.prompts import ChatPromptTemplate
from reasoning_engines.langchain_llm_wrappers import QuickAzureChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser 
from langchain.retrievers.you import YouRetriever


class InterviewResponseChain:
    def __init__(self, gpt_version='0613', temperature=0.5):
        self.gpt_version = gpt_version
        self.temperature = temperature
        self.interview_response_chain = self.get_interview_response_chain()
        self.yr = YouRetriever()
   

    def get_interview_response_chain(self):
        keyword_extraction_prompt = ChatPromptTemplate.from_template(
            """When presented with a question, take a moment to thoroughly consider its meaning. Then, identify {num_keywords} Google search keywords that are highly effective. These keywords should be selected with the aim of yielding Google search results that contain the most pertinent information related to the question.  
        
        Question:  
        {question}  
        
        Please provide the outcome in the form of a list of dictionaries, where each dictionary contains a single key called 'keyword' and its corresponding value is the relevant keyword."""
        )
        keyword_extraction_chain = keyword_extraction_prompt | QuickAzureChatOpenAI(self.gpt_version, temperature=self.temperature) | SimpleJsonOutputParser()    


        summarization_prompt = ChatPromptTemplate.from_template(
        """Upon being provided with a keyword and its associated context, you are requested to compose a concise and informative summary.  
        
        Keyword: {keyword}  
        
        Context:  
        {context}  
        
        Your summary should capture the essence of the context, emphasizing the relevance to the given keyword."""  
        )

        summarization_chain = RunnablePassthrough.assign(   
            summary = (RunnablePassthrough.assign(
            context = lambda x : ' '.join([document.page_content for document in self.yr.get_relevant_documents(x['keyword'])])) | summarization_prompt | QuickAzureChatOpenAI(self.gpt_version, temperature=self.temperature) )
        ) |  RunnableLambda(lambda x : 'Keyword: '+ x['keyword']+ '\nSummary: ' +x['summary'].content) 


        intermediate_chain = keyword_extraction_chain | summarization_chain.map() | RunnableLambda(lambda x : {'context':'\n\n'.join(x)})

        response_prompt = ChatPromptTemplate.from_template(
            """As the CEO of LG Electronics, overseeing the home appliance division, you have spearheaded the successful integration of generative AI technologies into the product line. This innovation has not only been met with market success but has also earned you prestigious recognition.  
        
        During the award ceremony, you are approached by reporters who pose the following question:  
        
        Question: {question}  
        
        To assist in crafting a well-informed response, your secretary has provided you with context containing key terms and their related summaries. Utilizing this context, along with the question provided, please deliver an answer to the reporters. Your response should be articulate, informative, and demonstrate a nuanced understanding of AI technology, including both its benefits and potential drawbacks. Furthermore, your answer should explore viable solutions to mitigate any negative implications.  
        
        Context:  
        {context}"""
        )
        response_chain =RunnablePassthrough.assign(
            context = lambda x: intermediate_chain.invoke({'question':x['question'], 'num_keywords':x['num_keywords']})
        )  | response_prompt | QuickAzureChatOpenAI(self.gpt_version, temperature=self.temperature) | StrOutputParser()

        return response_chain        

    def __call__(self, question, num_keywords):
        return self.interview_response_chain.invoke({'num_keywords':num_keywords, 'question':question})