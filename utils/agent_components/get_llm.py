import os
from utils.load_vars import get_param
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)


def get_base_llm(config):
    try: streaming = config.streaming.value
    except AttributeError: streaming = config.streaming  
    callbacks = [FinalStreamingStdOutCallbackHandler()] if streaming else []
    if config.provider == 'AzureChatOpenAI':
        return AzureChatOpenAI(deployment_name=get_param('azure_deployment_name'), model_name=get_param('azure_model_name'), temperature=0.0, streaming=streaming, callbacks=callbacks)
    elif config.provider == 'ChatOpenAI':
        # return ChatOpenAI(temperature=0.0, streaming=streaming, model_kwargs={'engine':get_param('model_name')})
        return ChatOpenAI(temperature=0.0, streaming=streaming, model_name=get_param('model_name'), model_kwargs={'engine':get_param('model_name')}, callbacks=callbacks)


'''
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                      max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"],
                                      temperature=temperature)
                                      #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""
'''