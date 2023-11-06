# Background
This prompt is designed to generate a dialogue line for a chatbot engaged in a conversation. It relies on two input variables, {persona} and {input}:
 - persona : This provides a description of the character that the language model (LLM) is expected to represent when generating the dialogue line. This persona description helps to inform the tone, style, and content of the chatbot's generations. An example looks like
```
persona = '{
    "gender": "male", 
    "name": "Joshua Ramos", 
    "language": "English", 
    "nationality": "Saudi_Arabia", 
    "age": 93, 
    "hobbies": ["traveling"], 
    "talkative": true, 
    "characteristics": ["Meticulous", "NotDiligent"], 
    "education_level": "master_degree_level"
    }'
```

 - input : This refers to the latest message from the conversation partner. If this input is an empty string, it is interpreted as a signal for the LLM to produce a dialogue line initiating a new conversation. An example looks like
```
input = "I need to wire money to my friend. But I don't know how to do it. I have no one to ask. Can you help me?"
```

The chatbot generates a response to the {input}.

# Use Cases
This prompt will be useful in synthesizing datasets of dialog lines. To that end, it has two main abilities:
- Often, a set of {persona} descriptions is generated automatically and randomly, rather than being manually crafted. This can sometimes lead to conceptual conflicts, where two or more characteristics contradict each other. This prompt therefore instructs the LLM to meticulously review the given {persona} to identify any such conflicts and make necessary modifications.

- This prompt also urges the LLM to polish the created dialog line through iterative review making sure the dialog line aligns with the modified {persona}.


# How to play around?
The following code snippet provides an example of how you can utilize the prompt with LLMs. You can swap out AzureChatOpenAI() for any LLM you want.
```
from langchain import hub
from langchain.chat_models import AzureChatOpenAI

def streaming_output(response_generator):
    for chunk in response_generator:
        print(chunk.content, end='')    

prompt = hub.pull("jet-taekyo-lee/persona-based-dialog")
chat_model = AzureChatOpenAI(deployment_name=LLM_DEPLOYMENT_NAME, temperature=0)
chain = prompt | chat_model

response = chain.stream({'persona':persona, 'input': 'Life is so boring. Why is that?'})
streaming_output(response)
```

Visit the provided [GitHub repo](https://github.com/Taekyo-Lee/LLM-powered-apps/tree/main/Custom_Prompts) and download [persona-based-dialog/nb_persona-based-dialog.ipynb](https://github.com/Taekyo-Lee/LLM-powered-apps/blob/main/Custom_Prompts/persona-based-dialog/nb_persona-based-dialog.ipynb) file.