import os
from langchain.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def bedrock_chatbot():
    bedrock_llm = Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2:1',
        model_kwargs={
            "prompt": "\n\nHuman:<prompt>\n\nAssistant:",
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "max_tokens_to_sample": 512
        }
    )

    return bedrock_llm

def buff_memory():
    buff_memory = bedrock_chatbot()
    memory = ConversationBufferMemory(llm = buff_memory, max_token_limit = 200)
    return memory

def cnvs_chain(input_text, memory):
    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm = chain_data, memory = memory, verbose = True)
    chat_reply = cnvs_chain.predict(input = input_text)
    return chat_reply
