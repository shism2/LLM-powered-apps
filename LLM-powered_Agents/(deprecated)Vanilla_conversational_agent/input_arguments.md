## Configuration

You can configure the application by passing input parameters that follow Python's argparse syntax. Since all input parameters have default values, you can run the application by just executing `python demo.py` if necessary environment variables are set in the '.env' file. However, you're welcome to manually provide specific arguments you want.


### Input parameters

--provider: Choose between 'AzureChatOpenAI' and 'ChatOpenAI'. The default is 'ChatOpenAI'.

--agent_type: Specify the agent type. Options include 'openai' and 'react'. The default is 'openai'.

--retrieval_chain_type: Define the retrieval chain type. Options include 'stuff' and 'map_reduce'. The default is 'stuff'.

--verbose: Toggle the display of detailed output messages. The default is 'true'.

--search_tool: Select the web-search tool. Options include 'YDC', 'Google', and 'Serp'. The default is 'YDC'.

--qna_log_folder: Specify the directory for Q&A logs. The default is 'loggers/qna_logs'.

--scratchpad_log_folder: Choose the directory for scratchpad logs. The default is 'loggers/scratchpad_logs'.

--streaming: Decide whether to stream the output. The default is 'true'.


Each argument should be passed in the format --argument value. For example, python demo.py --provider AzureChatOpenAI would set the provider to 'AzureChatOpenAI'.