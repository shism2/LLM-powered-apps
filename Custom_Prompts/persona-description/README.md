# Background
The prompt [jet-taekyo-lee/persona-based-dialog](https://smith.langchain.com/hub/jet-taekyo-lee/persona-based-dialog?organizationId=99c107dc-9be4-5ebc-9aee-41fafaa9d426) generates a dialogue line for a chatbot through a step-by-step reasoning process. The Language Learning Model (LLM) that uses this prompt is expected to review a given persona, refine it, and generate a detailed description of the refined persona before generating the final dialogue line.

**This prompt** is designed to provide intermediate artifacts as output. Its output is a JSON blob with three keys: 'original_persona', 'refined_persona', and 'refined_persona_description'. This approach is particularly useful when you need not only the final dialogue line but also an understanding of the factors that influenced it.


This prompt expects a single input variable, {persona}:
 - persona : This provides a structured description of the character that the LLM is expected to represent when generating the dialogue line. The persona description informs the tone, style, and content of the chatbot's responses. Here is an example:
```
persona = '{
    "gender": "unknown",
    "name": "Roy Kirby",
    "language": "English",
    "nationality": "Germany",
    "age": 28,
    "hobbies": ["hiking", "fishing", "painting"],
    "talkative": false,
    "characteristics": ["Timid", "NotDismissive", "Jovial", "Resourceful"],
    "education_level": "genius_level"
  }'
```

The chatbot refines the persona and writes a description of the refined persona. These, along with the original persona, are output in JSON format.
```
output = 
'{
  "original_persona": {
    "gender": "unknown",
    "name": "Roy Kirby",
    "language": "English",
    "nationality": "Germany",
    "age": 28,
    "hobbies": ["hiking", "fishing", "painting"],
    "talkative": false,
    "characteristics": ["Timid", "NotDismissive", "Jovial", "Resourceful"],
    "education_level": "genius_level"
  },
  "refined_persona": {
    "gender": "male",
    "name": "Roy Kirby",
    "language": "English",
    "nationality": "United_Kingdom",
    "age": 28,
    "hobbies": ["hiking", "fishing", "painting"],
    "talkative": false,
    "characteristics": ["Timid", "NotDismissive", "Jovial", "Resourceful"],
    "education_level": "genius_level"
  },
  "refined_persona_description": {
    "description_of_persona": "Roy Kirby is a 28-year-old male from the United Kingdom. He is a genius-level intellect who speaks English. Despite being timid, he is not dismissive and has a jovial and resourceful nature. He is not very talkative. His hobbies include hiking, fishing, and painting."
  }
}'
```

# Use Cases
This prompt is beneficial for creating datasets of high-quality personas without any contradictions.
- Often, a set of {persona} descriptions is generated automatically and randomly, rather than being manually crafted. This can sometimes lead to conceptual conflicts, where two or more characteristics contradict each other. This prompt therefore instructs the LLM to meticulously review the given {persona} to identify any such conflicts and make necessary modifications.

# OpenAI Function calling
- This prompt needs OpenAI's Function calling. To test it, you need to bind the 'functions' argument to the LLM. A Jupyter notebook file is provided below for you to try it out.
- Unfortunately, I found that currently, it raises an error when trying it on [LangChain Hub's Prompt Playground](https://smith.langchain.com/hub/jet-taekyo-lee/persona-description/playground?organizationId=99c107dc-9be4-5ebc-9aee-41fafaa9d426) even with the 'Function Call' parameter set. I am investigating the cause and will fix it as soon as possible. In the meantime, please try this prompt on your local machine using the provided notebook file.


# How to play around?
The following code snippet provides an example of how you can utilize the prompt with LLMs.  Due to OpenAI Function calling, you need to use OpenAI's LLMs.
```
from user_persona import UserPersona, get_random_user_persona
from user_persona import UserProfile
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain import hub
from langchain.chat_models import AzureChatOpenAI

persona = get_random_user_persona().json()
openai_function_profile = convert_pydantic_to_openai_function(UserProfile)
prompt = hub.pull("jet-taekyo-lee/persona-description")
chat_model = AzureChatOpenAI(deployment_name=LLM_DEPLOYMENT_NAME, temperature=0).bind(functions=[openai_function_profile], function_call={'name':'UserProfile'})

chain = prompt | chat_model
response = chain.invoke({'persona':persona})
print(response.additional_kwargs['function_call']['arguments'])
```

Visit the provided [GitHub repo](https://github.com/Taekyo-Lee/LLM-powered-apps/tree/main/Custom_Prompts/persona-description) and download [persona-description/nb_persona-description.ipynb](https://github.com/Taekyo-Lee/LLM-powered-apps/blob/main/Custom_Prompts/persona-description/nb_persona-description.ipynb) file.