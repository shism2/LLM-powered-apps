import os, load_envs, argparse, sys
import agent_enums
import gradio as gr
from loggers.qna_logger import get_qa_logger, logging_qa
from loggers.agent_scratchpad_logger import ScratchpadLogger, read_logs_from_file

from external_memories import SimpleListChatMemory
from configurations import Configurations
from langchain.chat_models import AzureChatOpenAI
from CustomAgents import ReActAgent


## argments
parser = argparse.ArgumentParser()
parser.add_argument("--agent_type", type=agent_enums.Agentype, choices=list(agent_enums.Agentype), default=agent_enums.Agentype.react)
parser.add_argument("--azure_d_name", type=agent_enums.AzureDeploymentName, choices=list(agent_enums.AzureDeploymentName), default=agent_enums.AzureDeploymentName.gpt4_32k)
parser.add_argument("--retrieval_chain_type", type=agent_enums.RetriecalChainType, choices=list(agent_enums.RetriecalChainType), default=agent_enums.RetriecalChainType.stuff)
parser.add_argument("--llm_search_api_chain_type", type=agent_enums.RetriecalChainType, choices=list(agent_enums.RetriecalChainType), default=agent_enums.RetriecalChainType.stuff)
parser.add_argument("--verbose", type=agent_enums.Boolean, choices=list(agent_enums.Boolean), default=agent_enums.Boolean.true)
parser.add_argument("--search_tool", type=agent_enums.Search, choices=list(agent_enums.Search), default=agent_enums.Search.YDC)
parser.add_argument("--qna_log_folder", type=str, default='loggers/qna_logs')
parser.add_argument("--scratchpad_log_folder", type=str, default='loggers/scratchpad_logs')
args = parser.parse_args()


## Create agent
config = Configurations(**vars(args))
llm = AzureChatOpenAI(deployment_name=config.azure_gpt_deployment_name.value, temperature=0.0)
agent = ReActAgent(llm, config=config)


## Folder-existing check
for k, v in config:
    if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
        os.makedirs(v)  


## Gradio functions
def qna(human_message, temperature, max_tokens):
    original_stdout = sys.stdout 
    sys.stdout =ScratchpadLogger(config.scratchpad_log_folder)  
    agent.set_temperature(temperature)        
    agent.set_max_tokens(max_tokens)        
    response = agent(human_message)    
    GUI_CHAT_RECORD(human_message, response)
    logging_qa(qa_logger, human_message, response)
    sys.stdout = original_stdout
    
    return GUI_CHAT_RECORD.chat_history




## Declear variables and objects
qa_logger = get_qa_logger(config.qna_log_folder)
GUI_CHAT_RECORD = SimpleListChatMemory()



## Gradio blcok
with gr.Blocks() as demo:  
    gr.Markdown(f"# ReAct agent")  
    chatbot_window = gr.Chatbot(height=450) #chatbot acts as chat history window
    with gr.Accordion(label="Settings", open=False):
        with gr.Row():
            with gr.Column():
                system_msg = gr.Textbox(label="System message", lines=20, max_lines=11, value=agent.system_msg, interactive=False)
            with gr.Column():
                with gr.Row():
                    qna_log_folder = gr.Textbox(label="QnA logs", value=config.qna_log_folder, interactive=False)
                    scratchpad_log_folder = gr.Textbox(label="scratchpad logs", value=config.scratchpad_log_folder, interactive=False)
                append_system_message = gr.Textbox(label="Append system message with", interactive=True)
                with gr.Row():
                    append_system_message_btn = gr.Button("Append system message")
                    reset_system_message_btn = gr.Button("Reset system message")
    
    append_system_message_btn.click(agent.append_sysem_msg, inputs=[append_system_message], outputs=[system_msg])
    reset_system_message_btn.click(agent.reset_system_msg, inputs=[], outputs=[system_msg])

    with gr.Row():
        with gr.Column():
            agent_scratchpad = gr.Textbox(label="ReAct scratchpad", lines=20, max_lines=11)
        with gr.Column():
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1, value=0.0, step=0.1)
            max_tokens = gr.Slider(label="Max tokens for completion", minimum=1, maximum=4096, value=2048, step=1)
            
            question = gr.Textbox(label="Question")
            
            with gr.Row():
                run_btn = gr.Button("Run")
                clr_screen = gr.Button("Clear screen")
                
                # with gr.Column():
                #     run_btn = gr.Button("Run")
                # with gr.Column():
                #     clr_screen = gr.Button("Clear screen")


    demo.load(read_logs_from_file, scratchpad_log_folder, agent_scratchpad, every=1)
    run_btn.click(qna, inputs=[question, temperature, max_tokens], outputs=[chatbot_window])
    clr_screen.click(GUI_CHAT_RECORD.clear_memory, inputs=[], outputs=[chatbot_window])



gr.close_all()    
demo.queue().launch(share=True)