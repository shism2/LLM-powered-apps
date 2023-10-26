import os
import load_envs
LLM_DEPLOYMENT_NAME= os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME= os.getenv("EMBEDDING_DEPLOYMENT_NAME")
from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(deployment_name=LLM_DEPLOYMENT_NAME, temperature=0.0)
from CustomAgents import OpenAIFunctionCallAgent, ZeroShotReActAgent
import gradio as gr
openai_agent = ZeroShotReActAgent(llm)



system_message = openai_agent.agent.llm_chain.prompt.messages[0].prompt.template

class ChatHistory:
    def __init__(self):
        self.chat_history = list()

    def __len__(self):
        return len(self.chat_history)

    def __call__(self, question, answer):
        self.chat_history.append((question, answer))

    def clear_memory(self):
        self.chat_history = list()


GUI_CHAT_HISTORY = ChatHistory()


def qna(human_message):
    answer = openai_agent.run(human_message)
    GUI_CHAT_HISTORY(human_message, answer)
    return GUI_CHAT_HISTORY.chat_history


with gr.Blocks() as demo:  
    gr.Markdown(f"# ReAct agent")  
    chatbot_window = gr.Chatbot(height=450) #chatbot acts as chat history window
    with gr.Row():
        with gr.Column():
            system_message = gr.Textbox(label="System message", lines=20, max_lines=11, value=system_message)
        with gr.Column():
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1, value=0.0, step=0.1)
            max_tokens = gr.Slider(label="Max tokens for completion", minimum=1, maximum=4096, value=2048, step=1)
            # with gr.Accordion(label="Tokens and price",open=False):
            with gr.Row():
                with gr.Column():
                    prompt_tokens = gr.Textbox(label="Tokens", value='Prompt: 0, Completion: 0')
                with gr.Column():
                    price = gr.Textbox(label="Price for the session (KRW)", value='0')
            question = gr.Textbox(label="Question")
            
            with gr.Row():
                with gr.Column():
                    btn = gr.Button("Run")
                with gr.Column():
                    clr_memory_and_run_btn = gr.Button("Clear memory & Run")
                with gr.Column():
                    example_btn = gr.Button("Create a sample question from llm")
                with gr.Column():
                    clr_memory_btn = gr.Button("Clear memory")

    btn.click(qna, inputs=[question], outputs=[chatbot_window])
    # example_btn.click(sample_question, inputs=[], outputs=[question])
    # clr_memory_btn.click(clear_memory, inputs=[], outputs=[])
    # clr_memory_and_run_btn.click(clear_memory_and_run, inputs=[system_message, temperature, max_tokens, question], outputs=[question, chatbot_window, prompt_tokens, price])


gr.close_all()    
demo.launch(share=True)