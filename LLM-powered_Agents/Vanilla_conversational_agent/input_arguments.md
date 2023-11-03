## Configuration

You can configure the application by passing input parameters that follow Python's argparse syntax. Since all input parameters have default values, you can run the application by jusy executing `python demo.py` if necessary environment variables are set in the '.env' file. However, you're welcome to manually provide specific arguments you want.


### Input parameters

You can modify configurations by adding arguments when executing `python demo.py`, adhering to Python's argparse syntax. The configurable input arguments are as follows:

--provider: Choose between 'AzureChatOpenAI' and 'ChatOpenAI'. The default is 'ChatOpenAI'.

--agent_type: Set the agent type. Choose between 'openai' and 'react'. The default is 'openai'.

--retrieval_chain_type: Define the retrieval chain type. Choose between 'stuff' and 'map_reduce'. The default is 'stuff'.

--verbose: Choose whether to display detailed output messages. The default is 'true'.

--search_tool: Select the web-search tool. Choose among ['YDC', 'Google', 'Serp'] The default is 'YDC'.

--qna_log_folder: Determine the directory for Q&A logs. The default is 'loggers/qna_logs'.

--scratchpad_log_folder: Choose the directory for scratchpad logs. The default is 'loggers/scratchpad_logs'.

--streaming: Decide whether to stream the output. The default is 'true'.

Each argument should be passed in the format --argument value. For example, python demo.py --provider AzureChatOpenAI would set the provider to 'AzureChatOpenAI'.