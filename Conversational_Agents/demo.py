import os, load_envs, argparse, sys
import agent_enums
import gradio as gr
from loggers.qna_logger import get_qa_logger, logging_qa
from loggers.agent_scratchpad_logger import ScratchpadLogger, read_logs_from_file

from external_memories import SimpleListChatMemory
from configurations import Configurations, folder_existence_check
from langchain.chat_models import AzureChatOpenAI
from CustomAgents import ReActAgent


## argments
parser = argparse.ArgumentParser()
parser.add_argument("--agent_type", type=agent_enums.Agentype, choices=list(agent_enums.Agentype), default=agent_enums.Agentype.openai)
parser.add_argument("--azure_d_name", type=agent_enums.AzureDeploymentName, choices=list(agent_enums.AzureDeploymentName), default=agent_enums.AzureDeploymentName.gpt4_32k)
parser.add_argument("--retrieval_chain_type", type=agent_enums.RetriecalChainType, choices=list(agent_enums.RetriecalChainType), default=agent_enums.RetriecalChainType.stuff)
parser.add_argument("--llm_search_api_chain_type", type=agent_enums.RetriecalChainType, choices=list(agent_enums.RetriecalChainType), default=agent_enums.RetriecalChainType.stuff)
parser.add_argument("--verbose", type=agent_enums.Boolean, choices=list(agent_enums.Boolean), default=agent_enums.Boolean.true)
parser.add_argument("--search_tool", type=agent_enums.Search, choices=list(agent_enums.Search), default=agent_enums.Search.YDC)
parser.add_argument("--qna_log_folder", type=str, default='loggers/qna_logs')
parser.add_argument("--scratchpad_log_folder", type=str, default='loggers/scratchpad_logs')
parser.add_argument("--streaming", type=agent_enums.Boolean, choices=list(agent_enums.Boolean), default=agent_enums.Boolean.true)
args = parser.parse_args()

## Create objects
config = Configurations(**vars(args))
folder_existence_check(config)
llms = [AzureChatOpenAI(deployment_name=config.azure_gpt_deployment_name.value, temperature=0.0, streaming=config.streaming.value)]    
agents = [ReActAgent(llms[0], config=config)]
qa_logger = get_qa_logger(config.qna_log_folder)
GUI_CHAT_RECORD = SimpleListChatMemory()


## Folder-existing check
for k, v in config:
    if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
        os.makedirs(v)  


## Gradio functions
def qna(human_message, temperature, max_tokens):
    global agents, llms, config
    try:
        if agents[0].config.agent_type.value == 'zeroshot react':
            gr.Warning('Memory for ReAct agent temporarily disabled.')
        original_stdout = sys.stdout 
        sys.stdout =ScratchpadLogger(config.scratchpad_log_folder)  
        agents[0].set_temperature(temperature)        
        agents[0].set_max_tokens(max_tokens)        
        response = agents[0](human_message)    
        GUI_CHAT_RECORD(human_message, response)
        logging_qa(qa_logger, human_message, response)
        sys.stdout = original_stdout
        return GUI_CHAT_RECORD.chat_history
    except Exception as e:
        raise gr.Error(e)

def agent_type_change(agent_type):
    global agents, llms
    config = Configurations(**vars(args))
    try:
        if agent_type == 'OpenAI Functions':
            config.agent_type = agent_enums.Agentype.openai
            gr.Info("Agent type chaged to OpenAI Functions. All chat history is deleted.")
        elif agent_type == 'ReAct':
            config.agent_type = agent_enums.Agentype.react
            gr.Info("Agent type chaged to ReAct. All chat history is deleted.")
        llms[0] = AzureChatOpenAI(deployment_name=config.azure_gpt_deployment_name.value, temperature=0.0, streaming=config.streaming.value)     
        agents[0] = ReActAgent(llms[0], config=config)
        GUI_CHAT_RECORD.clear_memory()
        return agents[0].system_msg, GUI_CHAT_RECORD.chat_history
    except Exception as e:
            raise gr.Error(e)

def append_system_message_func(msg):
    global agents, llms
    new_msg = agents[0].append_sysem_msg(msg)
    return new_msg

def reset_system_msg_func():
    global agents, llms
    original_msg = agents[0].reset_system_msg()
    return original_msg



## Gradio blcok
with gr.Blocks(title='Conversational Agent') as demo:  
    gr.Markdown(f"# Conversational Agent")  
    with gr.Row():
        agent_type_btn = gr.Radio(["OpenAI Functions", "ReAct"], label="Agent type", value="OpenAI Functions", interactive=True)
        llm_btn = gr.Radio(["gpt-3.5", "gpt-4", "llama2-7b", "llama2-13b"], label="LLM", value="gpt-4", interactive=False)
    
    chatbot_window = gr.Chatbot(height=450) #chatbot acts as chat history window
    with gr.Accordion(label="Settings", open=False):
        with gr.Row():
            with gr.Column():
                system_msg = gr.Textbox(label="System message", lines=20, max_lines=11, value=agents[0].system_msg, interactive=False, show_label=True, show_copy_button=True)
            with gr.Column():
                with gr.Row():
                    qna_log_folder = gr.Textbox(label="QnA logs", value=config.qna_log_folder, interactive=False, show_label=True, show_copy_button=True)
                    scratchpad_log_folder = gr.Textbox(label="scratchpad logs", value=config.scratchpad_log_folder, interactive=False, show_label=True, show_copy_button=True)
                append_system_message = gr.Textbox(label="Append system message with", interactive=True, show_label=True, show_copy_button=True)
                with gr.Row():
                    append_system_message_btn = gr.Button("Append system message")
                    reset_system_message_btn = gr.Button("Reset system message")
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1, value=0.0, step=0.1)
                with gr.Row():
                    max_tokens = gr.Slider(label="Max tokens for completion", minimum=1, maximum=4096, value=2048, step=1)
    
    agent_type_btn.change(agent_type_change, [agent_type_btn], [system_msg, chatbot_window])
    append_system_message_btn.click(append_system_message_func, inputs=[append_system_message], outputs=[system_msg])
    reset_system_message_btn.click(reset_system_msg_func, inputs=[], outputs=[system_msg])

    with gr.Row():
        with gr.Column():
            agent_scratchpad = gr.Textbox(label="Agent scratchpad", lines=20, max_lines=11, show_label=True, show_copy_button=True)
        with gr.Column():
            question = gr.Textbox(label="Question", show_label=True, show_copy_button=True)
            
            with gr.Row():
                run_btn = gr.Button("Run")
                clr_screen = gr.Button("Clear screen")
            with gr.Row(): 
                gr.Examples(
                    examples=["지금 몇시야?", "지금 뉴욕 날씨 어때?", "현재 기준으로 BTS의 최연장자의 나이를 log_10()에 넣으면 답이 뭐야?", 
                    "연립방정식 [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]*[x, y, z] = [1, -2, 0] 풀어줘", "아! 삶이란 무엇일까?",
                    "니가 사용할 수 있는 외부 툴들의 이름과 용도를 알려줘.", "안녕 내 이름은 구루야.", "내 이름이 뭐지?"],
                    inputs= [question],
                    label='Example questions'
                )

    run_btn.click(qna, inputs=[question, temperature, max_tokens], outputs=[chatbot_window])
    clr_screen.click(GUI_CHAT_RECORD.clear_memory, inputs=[], outputs=[chatbot_window])

    demo.load(read_logs_from_file, scratchpad_log_folder, agent_scratchpad, every=1)


gr.close_all()    
demo.queue().launch(share=True)