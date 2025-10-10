import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads,llm_with_tools,llm
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

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

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# ======================= Session Initialization ===================

# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Initialize chat_threads BEFORE anything that calls add_thread()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
    # If retrieve_all_threads returns None, ensure it's a list
    if st.session_state["chat_threads"] is None:
        st.session_state["chat_threads"] = []

# Initialize thread_id
if "thread_id" not in st.session_state:
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    add_thread(new_thread_id)

# Initialize thread title
if "thread_title" not in st.session_state:
    st.session_state["thread_title"] = "New Chat"


# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
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
    # if st.sidebar.button(str(thread_id)):
    #     st.session_state["thread_id"] = thread_id
    #     messages = load_conversation(thread_id)

    #     temp_messages = []
    #     for msg in messages:
    #         role = "user" if isinstance(msg, HumanMessage) else "assistant"
    #         temp_messages.append({"role": role, "content": msg.content})
    #     st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )