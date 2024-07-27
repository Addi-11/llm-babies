import streamlit as st
from openai import OpenAI
import os


st.title('🦜Ducky OpenAI Chat App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
os.environ['OPENAI_API_KEY'] = openai_api_key
client = OpenAI()

MESSAGES = "messages"
template = "You are a funny and sarcastic chatbot named DUCKY, always giving quirky answers to the user."
user_chat = []

def initialize_state():
    # intialize state
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = user_chat

    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')

    # loading chat for the session
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg["role"]).write(msg["content"])
        # each session re-runs the code, so need to load the chat again
        user_chat.append({"role":msg["role"], "content": msg["content"]})

def generate_response(input_text):
    user_chat.append({"role":"user", "content": input_text})
    # print(user_chat)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content":template}] + user_chat
    )
    response_msg = response.choices[0].message.content
    user_chat.append({"role":"assistant", "content": response_msg})

    return response_msg

def main():
    initialize_state()
    prompt: str = st.chat_input("Enter a prompt here")

    if prompt:
        st.session_state[MESSAGES].append({"role":"user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Please wait...."):
            response = generate_response(prompt)
            st.session_state[MESSAGES].append({"role":"assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
