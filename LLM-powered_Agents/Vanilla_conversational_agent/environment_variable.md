You are required to create a **.env** file in the **Vanilla_conversation_agent** folder. This file should contain specific environment variables. All environment-variable names are case-sensitive. Remember to replace all the '...........' placeholders with your actual API keys or values.

### 1. Environment variable for LLM
For **OpenAI**'s LLMs, please include these two variables:

OPENAI_API_KEY = '...........'

model_name = '...........' (For example, 'gpt-3.5-turbo-0613')

For **AzureOpenAI**'s LLMs, please include these four variables:

OPENAI_API_TYPE = 'azure' (This should always be 'azure')

OPENAI_API_VERSION = '...........'

OPENAI_API_BASE= '...........'

OPENAI_API_KEY = '...........'

deployment_name = '...........'


### 2. Environment variable for weather tools
OPENWEATHERMAP_API_KEY = '...........'


### 3. Environment variable for math tools
WOLFRAM_ALPHA_APPID = '...........'


### 4. Environment variable for web-search tools
There are three options for web searching, with You.com as the default. If you prefer to use the native Google Search API or Serp API instead of You.com, you should include `--search_tool Google` or `--search_tool Serp` when running `python demo.py`.

**Native Google search API**

GOOGLE_API_KEY = '...........'

GOOGLE_CSE_ID = '...........'

**Serp API**

SERPAPI_API_KEY = '...........'

**You.com API**

YDC_API_KEY = '...........'


