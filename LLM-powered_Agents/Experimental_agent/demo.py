# Path and environment variables
import os, sys
sys.path.extend(['..', '../..'])
from dotenv import load_dotenv
_ = load_dotenv('../../.env')

from utils.load_vars import get_param
import argparse
from utils.agent_components import agent_enums
import gradio as gr
from utils.loggings.qna_logger import get_qa_logger, logging_qa
from utils.loggings.agent_scratchpad_logger import ScratchpadLogger, read_logs_from_file
from utils.agent_components.external_memories import SimpleListChatMemory
from utils.agent_components.get_llm import get_base_llm
from utils.agent_components.configurations import Configurations, get_agent_type_enum
from agent_specific.customagents import ExperimentalAgent


## argments
parser = argparse.ArgumentParser()
parser.add_argument("--provider", choices=['AzureChatOpenAI', 'ChatOpenAI'], default='AzureChatOpenAI')
parser.add_argument("--agent_type", type=agent_enums.Agentype, choices=list(agent_enums.Agentype), default=agent_enums.Agentype.openai)
parser.add_argument("--retrieval_chain_type", type=agent_enums.RetrievalChainType, choices=list(agent_enums.RetrievalChainType), default=agent_enums.RetrievalChainType.stuff)
parser.add_argument("--api_retrieval_chain_type", type=agent_enums.RetrievalChainType, choices=list(agent_enums.RetrievalChainType), default=agent_enums.RetrievalChainType.stuff)
parser.add_argument("--verbose", type=agent_enums.Boolean, choices=list(agent_enums.Boolean), default=agent_enums.Boolean.true)
parser.add_argument("--search_tool", type=agent_enums.Search, choices=list(agent_enums.Search), default=agent_enums.Search.YDC)
parser.add_argument("--qna_log_folder", type=str, default='loggers/qna_logs')
parser.add_argument("--scratchpad_log_folder", type=str, default='loggers/scratchpad_logs')
parser.add_argument("--streaming", type=agent_enums.Boolean, choices=list(agent_enums.Boolean), default=agent_enums.Boolean.true)
parser.add_argument("--langsmith_project", type=str, default='RAG_style_agent_gradio_demo')
args = parser.parse_args()

## LangSmith Project
os.environ['LANGCHAIN_PROJECT'] = args.langsmith_project

## Create objects
config = Configurations(**vars(args))
agents = [ExperimentalAgent(config=config)]
qa_logger = get_qa_logger(config.qna_log_folder)
GUI_CHAT_RECORD = SimpleListChatMemory()


## Folder-existing check
for k, v in config:
    if (k.split('_folder')[-1]=='') and (not os.path.exists(v)):
        os.makedirs(v)  


## Gradio functions
def qna(human_message, temperature, max_tokens, max_token_none):
    global agents, config, GUI_CHAT_RECORD
    original_stdout = sys.stdout 
    try:
        sys.stdout =ScratchpadLogger(config.scratchpad_log_folder)  
        agents[0].set_temperature(temperature)    
        if max_token_none:    
            agents[0].set_max_tokens(None)
        else:
            agents[0].set_max_tokens(max_tokens)

        response, num_iter = agents[0](human_message, True)    
        GUI_CHAT_RECORD(human_message, response)
        logging_qa(qa_logger, human_message, response)
        sys.stdout = original_stdout
        return GUI_CHAT_RECORD.chat_history, '', num_iter
    except Exception as e:
        sys.stdout = original_stdout
        agents[0].delete_scratchpad_logs()
        with open(os.path.join(config.scratchpad_log_folder, 'scratch_log.log'), 'w') as file:  
            file.write('----- Error -----\n'+str(e))  
        raise gr.Error(e)

def agent_type_change(agent_type, provider):
    global agents
    try:
        config = Configurations(**vars(args))
        config.provider = provider        
        config.agent_type = get_agent_type_enum(agent_type)        
        agents[0] = ExperimentalAgent(config=config)
        GUI_CHAT_RECORD.clear_memory()
        gr.Info(f"Agent type chaged to {agent_type}. All chat history is deleted. System message is reset.")
        return agents[0].system_msg, GUI_CHAT_RECORD.chat_history
    except Exception as e:
            raise gr.Error(e)


def llm_change(provider, agent_type):
    global agents
    try:
        config = Configurations(**vars(args))
        config.provider = provider
        config.agent_type = get_agent_type_enum(agent_type)
       
        
        agents[0] = ExperimentalAgent(config=config)
        GUI_CHAT_RECORD.clear_memory()
        gr.Info(f"Provider changed to {provider}. All chat history is deleted. System message is reset.")
        return agents[0].system_msg, GUI_CHAT_RECORD.chat_history
    except Exception as e:
        raise gr.Error(e)


def clear_memory():
    global agents
    try:
        agents[0].agent_executor.memory.clear()
        gr.Info(f"Agent's memory has been reset. Agent forgets every conversation so far.")
    except Exception as e:
        raise gr.Error(e)

def append_system_message_func(msg):
    global agents
    new_msg = agents[0].append_sysem_msg(msg)
    return new_msg, ''

def reset_system_msg_func():
    global agents
    original_msg = agents[0].reset_system_msg()
    return original_msg, ''


def activation_max_token(max_token_none):
    if max_token_none:
        return gr.Slider.update(label="Max tokens for completion", minimum=1, maximum=4096, value=0, step=1, interactive=False)
        return 10
    else:
        return gr.Slider.update(label="Max tokens for completion", minimum=1, maximum=4096, value=2048, step=1, interactive=True)
        return 100

## Gradio blcok
with gr.Blocks(title='Conversational Agent') as demo:  
    gr.Markdown(f"# Conversational Agent")  
    with gr.Row():
        try: agent_type = agents[0].config.agent_type.value
        except AttributeError: agent_type = agents[0].config.agent_type
        agent_type_btn = gr.Radio(["OpenAI_Functions", "ReAct", "ReAct_RAG_style"], label="Agent type", value=agent_type, interactive=True)
        provider_btn = gr.Radio(["ChatOpenAI", "AzureChatOpenAI"], label="LLM", value=agents[0].config.provider, interactive=True)

    
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
                    with gr.Column():
                        max_tokens = gr.Slider(label="Max tokens for completion", minimum=1, maximum=4096, value=0, step=1, interactive=False)
                    with gr.Column():
                        max_token_none = gr.Checkbox(label="Remove the completion token limit", info="Allow as many completion tokens as your Tier allows", value=True, interactive=True)
    agent_type_btn.change(agent_type_change, [agent_type_btn, provider_btn], [system_msg, chatbot_window])
    provider_btn.change(llm_change, [provider_btn, agent_type_btn], [system_msg, chatbot_window])
    append_system_message_btn.click(append_system_message_func, inputs=[append_system_message], outputs=[system_msg, append_system_message])
    reset_system_message_btn.click(reset_system_msg_func, inputs=[], outputs=[system_msg, append_system_message])
    max_token_none.change(fn=activation_max_token, inputs=[max_token_none], outputs=[max_tokens])

    with gr.Row():
        with gr.Column():
            agent_scratchpad = gr.Textbox(label="Agent scratchpad", lines=20, max_lines=11, show_label=True, show_copy_button=True)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    num_iterations = gr.Textbox(label="Iterations", show_label=True, info='Number of iterations to get the final answer' )
                with gr.Column():
                    dummy = gr.Textbox(label="dummy", show_label=True)
            with gr.Row():
                question = gr.Textbox(label="Question", show_label=True, show_copy_button=True)
            
            with gr.Row():
                run_btn = gr.Button("Run")
                clr_screen = gr.Button("Clear screen")
                reset_memory = gr.Button("Reset memory")
            with gr.Row(): 
                gr.Examples(
                    examples=["지금 몇시야?", "지금 뉴욕 날씨 어때?", "현재 기준으로 BTS의 최연장자의 나이를 log_10()에 넣으면 답이 뭐야?", 
                    "연립방정식 [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]*[x, y, z] = [1, -2, 0] 풀어줘", "아! 삶이란 무엇일까?",
                    "니가 사용할 수 있는 외부 툴들의 이름과 용도를 알려줘.", "안녕 내 이름은 구루야.", "내 이름이 뭐지?"],
                    inputs= [question],
                    label='Example questions'
                )

    run_btn.click(qna, inputs=[question, temperature, max_tokens, max_token_none], outputs=[chatbot_window, question, num_iterations])
    clr_screen.click(GUI_CHAT_RECORD.clear_memory, inputs=[], outputs=[chatbot_window])
    reset_memory.click(clear_memory, inputs=[], outputs=[])

    demo.load(read_logs_from_file, scratchpad_log_folder, agent_scratchpad, every=1, queue=True)


gr.close_all()  
demo.queue().launch(share=True)
