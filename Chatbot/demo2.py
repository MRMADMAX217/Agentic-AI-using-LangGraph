import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def generate_thread_title(thread_id):
    conversation = load_conversation(thread_id)
    conversation_text = "\n".join([msg.content for msg in conversation])
    CONFIG = {'configurable': {'thread_id': thread_id}}
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=conversation_text + "\nGenerate a title by analysing the conversation and just return me the title only")]},config=CONFIG
    )
    messages = response.get("messages", [])
    if messages:
        title = messages[-1].content
    else:
        title = "Untitled"
    return title

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    
    # Temporary title
    st.session_state['thread_titles'][thread_id] = "New Chat"
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not

    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
# ------------------ Session setup ------------------
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Add first thread if it doesn't exist
if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])
    # Temporary title
    st.session_state['thread_titles'][st.session_state['thread_id']] = "New Chat"



# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # get the stored title for this thread
    thread_title = st.session_state['thread_titles'].get(thread_id, "Untitled")

    if st.sidebar.button(label=thread_title, key=f"thread_button_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages



# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    thread_id = st.session_state['thread_id']

    # Add user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': thread_id}}

    # Stream assistant response
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if st.session_state['thread_titles'].get(thread_id) == "New Chat":
        st.session_state['thread_titles'][thread_id] = generate_thread_title(thread_id)
