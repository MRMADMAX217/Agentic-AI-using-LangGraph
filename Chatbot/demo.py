import streamlit as st
import asyncio
# from langgraph_backend import chatbot
from demo_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Load history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

async def stream_response(user_input):
    placeholder = st.empty()
    full_response = ""

    async for event in chatbot.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG
    ):
        if "messages" in event:
            for msg in event["messages"]:
                if msg.type == "ai":
                    chunk = msg.content
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        # Run the async function inside Streamlit
        ai_message = asyncio.run(stream_response(user_input))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
