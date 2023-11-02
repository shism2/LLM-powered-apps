import os
from dotenv import load_dotenv
_ = load_dotenv('.env')

# # Azure OpenAI
# os.environ['OPENAI_API_TYPE'] =  os.getenv('OPENAI_API_TYPE')
# os.environ['OPENAI_API_VERSION'] = os.getenv("OPENAI_API_VERSION")
# os.environ['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# # Search
# ### Serp API
# os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")
# ### Google search API
# os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# ### You.com API
# os.environ['YDC_API_KEY'] = os.getenv("YDC_API_KEY")