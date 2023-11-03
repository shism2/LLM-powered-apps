# Vanilla Conversational Agent  
  
The Vanilla Conversational Agent is a foundational application that merges the functionality of a chatbot with the utility of external tools. It is designed to facilitate conversations with users by letting it carry out sophisticated tasks that often challenge traditional chatbots. This is achieved through the execution of external tools backed by logical reasoning steps. The reasoning mechanisms supported by this application include:
  
* ReAct (Shunyu Yao, et al.)  
* OpenAI Function   

## What kinds of external tools it can use? 

Currently, the tools this agent can use fall into five cateforeis:
- Time: Get current time
- Weather: Get current weather and weather prediction up to 7 days ahead
- Math: Sove complex math problems
- Web search: Do Google search
- Python REPL: Run Python code interactively while in conversation with chatbot 
  
  
## How to use? 

To taste Vanilla conversational agent, follow the three steps.
  
### Step 1: Copy contents
Copy the entire contents of in the forlder 'Vanilla_conversation_agent'. Alternatively, you can clone the entire repository 'https://github.com/Taekyo-Lee/LLM-powered-apps.git' and go into 'Vanilla_conversation_agent' in the cloned directory

### Step 2: Create a virtual environment and install dependencies
Create a virtual environment by using your favoriate method like python venv module or conda. And then install dependencies by using 'requirements.txt' file.

### Step 3: Create '.env' file and populate it with environment variables
For security reason, '.env' file is not pushed in this repository. You have to create it on your own and fill it with necessary environment variable. Read 'environment_variable.md' to get information about the required environment variables.

After doing all of the steps above, run 'python demo.py' on the terminal. You can also pass in various arguments when executing 'demo.py' by following Python's argparse library. Refer to 'environment_variable.md' to get information about arguments.mail.com. We are always looking to improve and welcome any feedback.   
  
