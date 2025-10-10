import streamlit as st
from langgraph_database_backend import chatbot, llm 
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

# def generate_thread_title(thread_id):
#     conversation=load_conversation(thread_id)
#     response=chatbot.invoke({'messages':[HumanMessage(content=conversation+"Generate a title by analysing the conversation ")]})
#     messages = response.get("messages", [])
#     if messages:
#         title = messages[-1].content
#     else:
#         title = "Untitled"
#     return title
from langchain_core.messages import HumanMessage

from langchain_core.messages import HumanMessage
import streamlit as st # Ensure st is imported

@st.cache_data(show_spinner=False)
def generate_thread_title(thread_id, conversation_len):
    conversation = load_conversation(thread_id)
    
    if not conversation:
        return "New Chat..."
        
    # Use only the first three turns for a title prompt
    conversation_text = "\n".join([msg.content for msg in conversation[:3]]) 
    
    title_prompt = "Generate a concise title (5 words max) for the following conversation and just return the title only:\n\n" + conversation_text
    
    # 💥 CHANGE: Use the direct LLM invoke instead of the full 'chatbot'
    response = llm.invoke([HumanMessage(content=title_prompt)]) 
    
    if response:
        # response is a BaseMessage, access content
        title = response.content.strip(' "') 
    else:
        title = "Untitled Conversation"
        
    return title



def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not

    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []

# if 'thread_id' not in st.session_state:
#     st.session_state['thread_id'] = generate_thread_id()

# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads'] = []

# if 'thread_title' not in st.session_state:
#     st.session_state['thread_title'] = generate_thread_title(st.session_state['thread_id'])
#     add_thread(st.session_state['thread_id'])
# **************************************** Session Setup ******************************
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    # Generate a new ID for the very first session
    new_thread_id = generate_thread_id()
    st.session_state['thread_id'] = new_thread_id
    # Add this initial thread to the list
    add_thread(new_thread_id) 

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# 'thread_title' state is fine, it will be regenerated in the sidebar as needed
if 'thread_title' not in st.session_state:
    st.session_state['thread_title'] = "New Chat" # Use a placeholder title

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Get the length of the conversation to invalidate the cache only when the chat grows
    current_conversation = load_conversation(thread_id)
    conversation_len = len(current_conversation)
    thread_title = generate_thread_title(thread_id, conversation_len) # Pass the length
    if st.sidebar.button(label=str(thread_title), key=f"thread_button_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    
    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        # 💡 Add the thread ID to top-level metadata for explicit LangSmith grouping
        'metadata': {'thread_id': str(st.session_state['thread_id'])} 
        }

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
