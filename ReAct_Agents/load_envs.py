import os
from dotenv import load_dotenv
_ = load_dotenv('.env')

os.environ['OPENAI_API_TYPE'] =  os.getenv('OPENAI_API_TYPE')
os.environ['OPENAI_API_VERSION'] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ['OPENAI_API_BASE'] = os.getenv("AZURE_OPENAI_API_ENDPOINT")
os.environ['OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")