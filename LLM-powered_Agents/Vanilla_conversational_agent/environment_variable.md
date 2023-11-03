You have to create **.env** file in **Vanilla_conversation_agent** folder. The below are the environment variables you write to **.env** file. All variables are case-sensitive.


### 1. Environment variable for LLM
If you use OpenAI, you need two environment variables:

OPENAI_API_KEY = '...........'

model_name = '...........' ## example is 'gpt-3.5-turbo-0613'

If you use OpenAI, you need four environment variables:

OPENAI_API_TYPE = 'azure' ## This should be fixed to 'azure'

OPENAI_API_VERSION = '...........'

OPENAI_API_BASE= '...........'

OPENAI_API_KEY = '...........'

deployment_name = '...........'


### 1. Environment variable for weather tools
OPENWEATHERMAP_API_KEY = '...........'


### 2. Environment variable for math tools
WOLFRAM_ALPHA_APPID = '...........'


### 3. Environment variable for web-search tools
You can choose one out of three options for web search. The default is using **You.com**. So if you only have api key for native Google search API or Serp API rather than You.com api, you need to pass add `--search_tool Google` or `--search_tool Serp` when running `python demo.py`.

**Native Google search API**

GOOGLE_API_KEY = '...........'

GOOGLE_CSE_ID = '...........'

**Serp API**

SERPAPI_API_KEY = '...........'

**You.com API**

YDC_API_KEY = '...........'


