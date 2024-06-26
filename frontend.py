import streamlit as st
import backend as be

st.title("Amazon Bedrock Chatbot")
st.session_state.memory = be.buff_memory()
st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message["text"])

input_text = st.chat_input("질문을 입력하세요.")
if input_text:
    with st.chat_message("나"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role":"user", "text":input_text})

    chat_response = be.cnvs_chain(input_text=input_text, memory=st.session_state.memory)

    with st.chat_message("챗봇"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role":"assistant", "text":chat_response})
