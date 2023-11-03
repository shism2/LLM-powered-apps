# Vanilla Conversational Agent  
  
The Vanilla Conversational Agent is a foundational application that merges the functionality of a chatbot with the utility of external tools. It is designed to facilitate conversations with users by letting it carry out sophisticated tasks that often challenge traditional chatbots. This is achieved through the execution of external tools backed by logical reasoning steps. The reasoning mechanisms supported by this application include:
  
* ReAct (Shunyu Yao, et al.)  
* OpenAI Function   

## What kinds of external tools it can use? 

The tools currently compatible with this agent fall into five categories:
- Time: Fetches current time
- Weather: Retrieves current weather and forecasts for up to 7 days
- Math: Solves complex math problems
- Web search: Executes Google searches
- Python REPL: Runs Python code interactively during chatbot conversations
  
  
## How to use? 

To experience the Vanilla Conversational Agent, follow these three steps:
  
**Step 1: Copy contents**
Copy all contents from the 'Vanilla_conversation_agent' folder. Alternatively, clone the entire repository from 'https://github.com/Taekyo-Lee/LLM-powered-apps.git' and navigate to the 'Vanilla_conversation_agent' directory in the cloned repository.

**Step 2: Create a virtual environment and install dependencies**
Create a virtual environment using your preferred method, such as the Python venv module or Conda. Next, install the necessary dependencies using the 'requirements.txt' file.

**Step 3: Create '.env' file and populate it with environment variables**
For security reasons, the '.env' file is not included in this repository. You'll need to create this file yourself and populate it with the required environment variables. Refer to 'environment_variable.md' for information about these variables.

After completing the above steps, run 'python demo.py' in your terminal. If you wish to pass various arguments when executing 'demo.py', follow the guidelines provided by Python's argparse library. For more information about arguments, refer to 'environment_variable.md'.
  
